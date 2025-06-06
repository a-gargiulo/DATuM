"""Define a class to load and probe the as-designed BeVERLI geometry."""

import sys
from typing import cast, List, Literal, Tuple, Optional

import numpy as np
import scipy.io as scio
import scipy.optimize as spoptimize
import trimesh
from .my_types import AnalyticGeometry, CadGeometry, FloatOrArray, HillGeometry

# Constants
HILL_WIDTH = 0.93472
HILL_HEIGHT = HILL_WIDTH / 5.0
HILL_CYL_SECTION_WIDTH = HILL_WIDTH / 10.0


class Beverli:
    """
    Create an object containing the BeVERLI Hill as-designed geometry.

    Both a CAD model or an analytic pointcloud can be loaded. Additionally, various probing methods are provided.
    """

    def __init__(self, orientation: Optional[float] = None, use_cad: bool = True):
        """Initialize attributes and load geometry."""
        self.width: float = HILL_WIDTH
        self.height: float = HILL_HEIGHT
        self.cyl_section_width: float = HILL_CYL_SECTION_WIDTH
        self.polynomial_coefficients: np.ndarray = self._calculate_polynomial_coefficients()
        self.use_cad: bool = use_cad
        self.mesh: HillGeometry = self._load_geometry()
        self.orientation: float = 0.0 if orientation is None else orientation
        self.rotate(self.orientation)

    def calculate_perimeter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the x1 and x3 coordinates of the hill perimeter at a specific orientation."""
        orientation_rad = np.deg2rad(self.orientation)

        corners = self._calculate_perimeter_corners(n_pts=1000)
        edges = self._calculate_perimeter_edges(n_pts=100)

        x1 = np.concatenate([item for i in range(4) for item in (edges[i][0], corners[i][0])])
        x3 = np.concatenate([item for i in range(4) for item in (edges[i][2], corners[i][2])])

        x1rot = x1 * np.cos(orientation_rad) + x3 * np.sin(orientation_rad)
        x3rot = -x1 * np.sin(orientation_rad) + x3 * np.cos(orientation_rad)

        return x1rot, x3rot

    def calculate_x1_x2(self, x3: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate a cross-sectional x1-x2 profile at specific orientation and x3 location."""
        x1_perim, x3_perim = self.calculate_perimeter()

        try:
            x1_ip = self._find_perimeter_intersection_points(x1_perim, x3_perim, x3)
        except ValueError as e:
            raise RuntimeError("Intersection point search failed.") from e
        x1_ip = np.sort(x1_ip)

        n_pts = 1000
        x1 = np.linspace(x1_ip[0], x1_ip[1], n_pts)
        x2 = np.zeros((n_pts,))
        for i in range(n_pts):
            x2[i] = self.probe_hill(x1[i], x3)

        x1 = np.append(x1, [6.0])
        x2 = np.append(x2, [0.0])
        x1 = np.concatenate(([-6.0], x1))
        x2 = np.concatenate(([0.0], x2))

        return x1, x2

    def probe_hill(self, x1: float, x3: float) -> float:
        """Find x2 at a specific (x1, x3) point."""
        if self.use_cad:
            try:
                if x1 < -1:
                    return 0
                else:
                    return self._probe_cad_hill(x1, x3)
            except ValueError as e:
                print(f"[ERROR]: {e}")
                sys.exit(-1)
        return self._probe_analytic_hill(x1, x3)

    def get_surface_normal_symmetric_hill_centerline(self, x1: float) -> np.ndarray:
        """
        Calculate the surface normal at a specific x1 location along the centerline at symmetric orientations.

        :param x1: The x1 location along the hill centerline, measured in meters.

        :return: The normal vector.
        :rtype: numpy.ndarray
        """
        if self.orientation not in [0, 45, 90, 135, 180, 225, 270, 315]:
            raise ValueError("Non-symmetric hill orientations are not supported.")

        # resolution
        dx1 = 1e-12

        slope = np.array([1, (self.probe_hill(x1 + dx1, 0) - self.probe_hill(x1, 0)) / dx1, 0])
        nvec = np.array([-slope[1], slope[0], 0])
        nvec = nvec / np.sqrt(nvec[0] ** 2 + nvec[1] ** 2 + nvec[2] ** 2)

        return nvec

    def rotate(self, rot_angle: float):
        """Rotate geometry about x2 axis (counterclockwise)."""
        if self.use_cad:
            self._rotate_cad(rot_angle)
        else:
            self._rotate_analytic(rot_angle)

    def _load_geometry(self) -> HillGeometry:
        """Load geometry from CAD or analytic model."""
        loader = self._load_cad if self.use_cad else self._load_analytic
        return loader()

    def _load_cad(self) -> CadGeometry:
        """Load geometry from CAD model."""
        file_path = "./datum/resources/geometry/cad/BeVERLI_Hill_Surface.stl"
        geometry = cast(trimesh.Trimesh, trimesh.load(file_path))
        return geometry

    def _load_analytic(self) -> AnalyticGeometry:
        """Load geometry from analytic model."""
        file_path = "./datum/resources/geometry/analytic/BeVERLI_Hill.mat"
        geometry = scio.loadmat(file_path)
        return geometry

    def _rotate_cad(self, rot_angle: float):
        """Rotate CAD geometry about x2 axis (counterclockwise)."""
        rot_angle_rad = np.deg2rad(rot_angle)
        rot_axis = np.array([0, 1, 0])
        rot_mat = trimesh.transformations.rotation_matrix(rot_angle_rad, rot_axis)
        cast(trimesh.Trimesh, self.mesh).apply_transform(rot_mat)

    def _rotate_analytic(self, rot_angle: float):
        """Rotate analytic geometry about x2 axis (counterclockwise)."""
        rot_angle_rad = np.deg2rad(rot_angle)

        geometry = cast(AnalyticGeometry, self.mesh)

        x1rot = np.cos(rot_angle_rad) * geometry["X"] + np.sin(rot_angle_rad) * geometry["Z"]
        x2rot = geometry["Y"]
        x3rot = -np.sin(rot_angle_rad) * geometry["X"] + np.cos(rot_angle_rad) * geometry["Z"]

        geometry["X"] = x1rot
        geometry["Y"] = x2rot
        geometry["Z"] = x3rot

    def _calculate_polynomial_coefficients(self) -> np.ndarray:
        """Calculate the fifth-degree polynomial coefficients."""
        xvec = [self.cyl_section_width / 2.0, self.width / 2.0]
        bvec = np.array([[self.height], [0], [0], [0], [0], [0]])
        amat = np.array(
            [
                [xvec[0] ** 5, xvec[0] ** 4, xvec[0] ** 3, xvec[0] ** 2, xvec[0], 1],
                [5 * xvec[0] ** 4, 4 * xvec[0] ** 3, 3 * xvec[0] ** 2, 2 * xvec[0], 1, 0],
                [20 * xvec[0] ** 3, 12 * xvec[0] ** 2, 6 * xvec[0], 2, 0, 0],
                [xvec[1] ** 5, xvec[1] ** 4, xvec[1] ** 3, xvec[1] ** 2, xvec[1], 1],
                [5 * xvec[1] ** 4, 4 * xvec[1] ** 3, 3 * xvec[1] ** 2, 2 * xvec[1], 1, 0],
                [20 * xvec[1] ** 3, 12 * xvec[1] ** 2, 6 * xvec[1], 2, 0, 0],
            ]
        )
        return np.linalg.solve(amat, bvec).reshape((6,))

    def _p5(self, x1: FloatOrArray) -> FloatOrArray:
        """Probe the fifth-degree polynomial profile."""
        return (
            self.polynomial_coefficients[0] * x1**5
            + self.polynomial_coefficients[1] * x1**4
            + self.polynomial_coefficients[2] * x1**3
            + self.polynomial_coefficients[3] * x1**2
            + self.polynomial_coefficients[4] * x1
            + self.polynomial_coefficients[5]
        )

    def _probe_corner(
        self, up: FloatOrArray, vp: FloatOrArray, sign_x1: Literal[-1, 1], sign_x3: Literal[-1, 1]
    ) -> Tuple[FloatOrArray, FloatOrArray, FloatOrArray]:
        """
        Construct one three-dimensional corner.

        This function uses the analytic parametrization in a specified quadrant.
        """
        x1 = (sign_x1 * self.cyl_section_width / 2) + (up - self.cyl_section_width / 2) * (
            abs(np.cos(vp)) ** 0.5
        ) * np.sign(np.cos(vp))

        x2 = self._p5(up)

        x3 = (sign_x3 * self.cyl_section_width / 2) + (up - self.cyl_section_width / 2) * (
            abs(np.sin(vp)) ** 0.5
        ) * np.sign(np.sin(vp))

        return x1, x2, x3

    def _calculate_flat_top(
        self, up: FloatOrArray, vp: FloatOrArray
    ) -> Tuple[FloatOrArray, FloatOrArray, FloatOrArray]:
        """Construct the flat top of the hill using the analytic parametrization."""
        x1 = up
        x2 = self.height
        x3 = vp
        return x1, x2, x3

    def _calculate_cyl_surface(
        self,
        up: FloatOrArray,
        vp: FloatOrArray,
        quadrant: Literal[1, 2, 3, 4],
    ) -> Tuple[FloatOrArray, FloatOrArray, FloatOrArray]:
        """
        Construct the cylindrical section of the hill.

        Uses the analytic parametrization in a specified quadrant.
        """
        if quadrant == 1:
            x1 = up
            x2 = self._p5(up)
            x3 = vp
        elif quadrant == 2:
            x1 = up
            x2 = self._p5(vp)
            x3 = vp
        elif quadrant == 3:
            x1 = -up
            x2 = self._p5(up)
            x3 = vp
        elif quadrant == 4:
            x1 = up
            x2 = self._p5(vp)
            x3 = -vp

        return x1, x2, x3

    def _probe_analytic_hill(self, x1p: float, x3p: float) -> float:
        """Probe the analytic hill geometry using a single probe."""
        # Rotate coordinate system counterclockwise to align with the 0 deg hill orientation
        orientation_rad = np.deg2rad(self.orientation)
        x1 = x1p * np.cos(orientation_rad) - x3p * np.sin(orientation_rad)
        x2 = None
        x3 = x1p * np.sin(orientation_rad) + x3p * np.cos(orientation_rad)

        # Calculate hill height
        def corner_sys_of_eqns(xvar, sign_x1, sign_x3):
            x1_corner, _, x3_corner = self._probe_corner(xvar[0], xvar[1], sign_x1, sign_x3)
            return [x1 - x1_corner, x3 - x3_corner]

        def solve_corner(sys_of_eqns, init_guess, sign_x1, sign_x3):
            maxiter = 6
            angle_increment_rad = 15 * np.pi / 180
            xx_2 = None
            i = 0
            for i in range(maxiter):
                solution = spoptimize.fsolve(sys_of_eqns, init_guess, args=(sign_x1, sign_x3))
                xx_2 = self._p5(solution[0])
                if 0 <= xx_2 <= self.height:
                    break
                angle_adjustment = (-1) ** i * (1 + i // 2) * angle_increment_rad
                init_guess = np.array([init_guess[0], init_guess[1] + angle_adjustment])
            return xx_2

        if (x1 > self.cyl_section_width / 2) and (x3 > self.cyl_section_width / 2):
            x2 = solve_corner(corner_sys_of_eqns, np.array([x1, np.pi / 4]), 1, 1)
        elif (x1 < -self.cyl_section_width / 2) and (x3 > self.cyl_section_width / 2):
            x2 = solve_corner(corner_sys_of_eqns, np.array([-x1, 3 * np.pi / 4]), -1, 1)
        elif (x1 < -self.cyl_section_width / 2) and (x3 < -self.cyl_section_width / 2):
            x2 = solve_corner(corner_sys_of_eqns, np.array([-x1, 5 * np.pi / 4]), -1, -1)
        elif (x1 > self.cyl_section_width / 2) and (x3 < -self.cyl_section_width / 2):
            x2 = solve_corner(corner_sys_of_eqns, np.array([x1, 7 * np.pi / 4]), 1, -1)
        elif (x1 > self.cyl_section_width / 2) and (-self.cyl_section_width / 2 <= x3 <= self.cyl_section_width / 2):
            x2 = self._p5(x1)
        elif (x1 < -self.cyl_section_width / 2) and (-self.cyl_section_width / 2 <= x3 <= self.cyl_section_width / 2):
            x2 = self._p5(-x1)
        elif (x3 < -self.cyl_section_width / 2) and (-self.cyl_section_width / 2 <= x1 <= self.cyl_section_width / 2):
            x2 = self._p5(-x3)
        elif (x3 > self.cyl_section_width / 2) and (-self.cyl_section_width / 2 <= x1 <= self.cyl_section_width / 2):
            x2 = self._p5(x3)
        elif (-self.cyl_section_width <= x1 <= self.cyl_section_width / 2) and (
            -self.cyl_section_width <= x3 <= self.cyl_section_width / 2
        ):
            x2 = self.height

        return cast(float, x2)

    def _probe_cad_hill(self, x1: float, x3: float) -> float:
        """Probe the CAD geometry at a single location using ray tracing."""
        # Define a ray with starting point and direction
        ray_origin = [x1, -0.1, x3]
        ray_direction = [0, 1, 0]

        geometry = cast(trimesh.Trimesh, self.mesh)

        intersections, _, _ = geometry.ray.intersects_location(ray_origins=[ray_origin], ray_directions=[ray_direction])

        if len(intersections) == 0:
            raise ValueError("No ray intersection found.\n")

        max_height_intersection = max(intersections, key=lambda point: point[1])
        height = max_height_intersection[1]

        return height

    def _calculate_perimeter_corners(self, n_pts: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Calculate the superelliptic corners of the perimeter."""
        corners = []
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 4, endpoint=False)):
            x1_corner, _, x3_corner = self._probe_corner(
                self.width / 2,
                np.linspace(angle, angle + np.pi / 2, n_pts),
                (-1) ** (i + (i + 1) // 3),
                (-1) ** (i // 2),
            )
            corners.append((x1_corner, _, x3_corner))
        return corners

    def _calculate_perimeter_edges(self, n_pts: int) -> List[Tuple[np.ndarray, int, np.ndarray]]:
        """Calculate the straight edges of the hill perimeter."""
        edges = []
        sign = -1
        for i in range(4):
            if i % 2 == 0:
                sign = -sign
                x1_edge = sign * self.width / 2 * np.ones((n_pts - 2,))
                x3_edge = np.linspace(
                    -sign * self.cyl_section_width / 2,
                    sign * self.cyl_section_width / 2,
                    n_pts,
                )[1:-1]
            else:
                x1_edge = np.linspace(
                    sign * self.cyl_section_width / 2,
                    -sign * self.cyl_section_width / 2,
                    n_pts,
                )[1:-1]
                x3_edge = sign * self.width / 2 * np.ones((n_pts - 2,))
            edges.append((x1_edge, 0, x3_edge))
        return edges

    @staticmethod
    def _find_perimeter_intersection_points(x1: np.ndarray, x3: np.ndarray, x3_ip: float) -> np.ndarray:
        """Find intersection points of hill's perimeter with the x1-x2-plane at a desired probe point."""
        # Find crossing points
        sign_x3 = np.sign(x3 - x3_ip)
        diff_sign_x3 = np.diff(sign_x3)
        ip_idxs = np.where(diff_sign_x3 != 0)
        # Check if the first and last element build a crossing point
        if (sign_x3[0] - sign_x3[-1]) != 0:
            ip_idxs = np.append(ip_idxs, len(x3) - 1, 0)
        ip_idxs = np.unique(ip_idxs)
        # Check also that only 2 intersection points exist
        if len(ip_idxs) > 2:
            raise ValueError("There should be only 2 crossing points.")

        x1_ip = np.zeros((2,))
        x1_ip[0] = np.interp(
            0,
            np.array([x3[ip_idxs[0]], x3[ip_idxs[0] + 1]]),
            np.array([x1[ip_idxs[0]], x1[ip_idxs[0] + 1]]),
        )
        x1_ip[1] = np.interp(
            0,
            np.array([x3[ip_idxs[1]], x3[ip_idxs[1] + 1]]),
            np.array([x1[ip_idxs[1]], x1[ip_idxs[1] + 1]]),
        )

        return x1_ip
