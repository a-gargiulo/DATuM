"""This module defines the :py:class:`datum.beverli.Beverli` class, which provides
essential methods for querying the BeVERLI Hill geometry."""
import sys
from typing import Dict, List, Tuple, Union, Literal

import numpy as np
import scipy.io as scio
import scipy.optimize as spoptimize
import trimesh

from . import utility
from .parser import InputFile


class Beverli:
    """Instances of the :py:class:`datum.beverli.Beverli` class encapsulate the
    complete as-designed BeVERLI Hill geometry.

    Each instance of this class is endowed with probing and visualization capabilities.

    :ivar orientation: Hill orientation, measured in degrees.
    :ivar cyl_section_width_m: Width of the hill's cylindrical section in units of
        meters.
    :ivar height_m: Height of the hill in units of meters.
    :ivar geometry: The hill geometry as `Trimesh <https://trimsh.org/trimesh.html>`_
        object or dictionary of 2-D NumPy ndarrays of shape (n, m), where n and m
        represent the number of available point cloud point in the x:sub:`1`- and
        x:sub:`2`- direction.
    :ivar polynomial_coefficients: 1-D NumPy ndarray of shape (6, ) containing the
        hill's fifth-degree polynomial coefficients.
    :ivar width_m: Hill width in units of meters.
    """

    def __init__(self) -> None:
        """Initialize a new instance of the :py:class:`datum.beverli.Beverli` class."""
        input_data = InputFile().data
        self.orientation = input_data["general"]["hill_orientation"]
        self.geometry_type = input_data["hill_geometry"]["type"]
        self.width_m: float = 0.93472
        self.height_m: float = 0.186944
        self.cyl_section_width_m: float = 0.093472
        self.polynomial_coefficients: np.ndarray = (
            self._compute_polynomial_coefficients()
        )
        self.geometry: Union[
            trimesh.Trimesh, Dict[str, np.ndarray]
        ] = self._load_hill_geometry()

    def compute_perimeter(self, hill_angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the hill's perimeter.

        :param hill_angle_deg: Yaw angle orientation of the hill, measured in degrees.
        :return: A tuple containing two 1-D NumPy ndarrays with shape (n, )
            representing the x:sub:`1` and x:sub:`3` coordinates of the hill,
            measured in meters, where n denotes the number of discrete perimeter
            points.
        """
        angle_rad = hill_angle_deg * np.pi / 180
        num_of_pts = 1000

        # Calculate corners
        corners = self._compute_corners_perimeter(num_of_pts)

        # Calculate edges
        edges = self._calculate_edges_perimeter(num_of_pts)

        # Concatenate perimeter
        x1_perimeter = np.concatenate(
            (
                edges[0][0],
                corners[0][0],
                edges[1][0],
                corners[1][0],
                edges[2][0],
                corners[2][0],
                edges[3][0],
                corners[3][0],
            )
        )
        x3_perimeter = np.concatenate(
            (
                edges[0][2],
                corners[0][2],
                edges[1][2],
                corners[1][2],
                edges[2][2],
                corners[2][2],
                edges[3][2],
                corners[3][2],
            )
        )

        # Rotate perimeter
        x1_perimeter_rotated = x1_perimeter * np.cos(angle_rad) + x3_perimeter * np.sin(
            angle_rad
        )
        x3_perimeter_rotated = -x1_perimeter * np.sin(
            angle_rad
        ) + x3_perimeter * np.cos(angle_rad)

        return x1_perimeter_rotated, x3_perimeter_rotated

    def compute_x1_x2_profile(self, x_3_m: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a cross-sectional x :sub:`1`-x:sub:`2` profile of the hill at a
        specific x:sub:`3` location.

        :param x_3_m: Location along the x:sub:3-direction of the cross-sectional profile,
            measured in meters.
        :return: A tuple containing two 1-D NumPy ndarrays with shape (n, )
            representing the x:sub:`1` and x:sub:`2` coordinates of the cross-sectional
            profile, measured in meters, where n denotes the number of discrete profile
            points.
        """
        num_of_pts = 1000
        hill_angle_deg = self.orientation
        x_1_perimeter, x_3_perimeter = self.compute_perimeter(hill_angle_deg)

        x_1_intersect = self._find_perimeter_intersection_points(
            x_1_perimeter, x_3_perimeter, x_3_m
        )
        x_1_intersect = np.sort(x_1_intersect)

        x_1_profile = np.linspace(x_1_intersect[0], x_1_intersect[1], num_of_pts)
        x_2_profile = np.zeros((num_of_pts,))
        for i in range(num_of_pts):
            if self.geometry_type == "CAD":
                x_2_profile[i] = self._probe_cad_hill(x_1_profile[i], x_3_m)
            else:
                x_2_profile[i] = self._probe_analytic_hill(
                    x_1_profile[i], x_3_m, hill_angle_deg
                )

        return x_1_profile, x_2_profile

    def probe_hill(self, x_1_m: float, x_3_m: float) -> float:
        """Probe the hill height at a specific (x:sub:`1`, x:sub:`3`) point.

        :param x_1_m: The probe's x:sub:`1` coordinate measured in meters.
        :param x_3_m: The probe's x:sub:`3` coordinate measured in meters.
        :return: Height of the hill at the specified probe's location.
        """

        if self.geometry_type == "CAD":
            return self._probe_cad_hill(x_1_m, x_3_m)
        return self._probe_analytic_hill(
            x_1_m, x_3_m, self.orientation
        )

    def get_surface_normal_symmetric_hill_centerline(self, x_1_m: float) -> np.ndarray:
        """Compute the hill surface normal along the hill's centerline at a specific
        x:sub:`1` location.

        :param x_1_m: The x:sub:`1` coordinate of the profile to extract.
        :return: A NumPy ndarray of shape (3, ) containing the normal vector components.
        """
        try:
            if self.orientation not in [0, 45, 90, 135, 180, 225, 270, 315]:
                raise ValueError("Non-symmetric hill orientation not supported.")
        except ValueError as err:
            print(f"ERROR: {err}\n")
            sys.exit(1)

        # resolution
        dx1 = 1e-12

        slope = np.array(
            [1, (self.probe_hill(x_1_m + dx1, 0) - self.probe_hill(x_1_m, 0)) / dx1, 0]
        )
        nvec = np.array([-slope[1], slope[0], 0])
        nvec = nvec / np.sqrt(nvec[0] ** 2 + nvec[1] ** 2 + nvec[2] ** 2)

        return nvec

    # Protected methods
    def _load_hill_geometry(
        self,
    ) -> Union[trimesh.Trimesh, Dict[str, np.ndarray]]:
        """Load the hill geometry from the CAD or analytical model.

        :return: The hill geometry as a `Trimesh <https://trimsh.org/trimesh.html>`_
            object or a dictionary of 1-D NumPy ndarrays of shape (n, ) representing
            the geometry's Cartesian coordinates, where n denotes the number of
            available discrete points of the corresponding 3-D geometry point cloud.
        """
        if self.geometry_type == "CAD":
            return self._load_cad_hill()
        return self._load_analytic_hill()

    def _load_cad_hill(self) -> trimesh.Trimesh:
        """Load the hill geometry from the CAD model.

        :return: The hill geometry as a `Trimesh <https://trimsh.org/trimesh.html>`_
            object.
        """
        input_data = InputFile().data
        # STL file path
        stl_file_path = utility.find_file(
            input_data["system"]["geometry_data_root_folder"],
            input_data["hill_geometry"]["cad_geometry"],
        )

        hill_geometry = trimesh.load(stl_file_path)

        # Rotate hill and return
        return self._rotate_cad_hill(hill_geometry)

    def _load_analytic_hill(self) -> Dict[str, np.ndarray]:
        """Load the hill geometry from the analytic model.

        :return: The hill geometry as a dictionary of 1-D NumPy ndarrays of shape (n, )
            representing the geometry's Cartesian coordinates, where n denotes the
            number of available discrete points of the corresponding 3-D geometry point
            cloud.
        """
        input_data = InputFile().data
        # Construct analytic geometry path
        mat_file_path = utility.find_file(
            input_data["system"]["geometry_data_root_folder"],
            input_data["hill_geometry"]["analytic_geometry"],
        )

        # Load analytic geometry
        hill_geometry = scio.loadmat(mat_file_path)

        # Rotate and return geometry
        return self._rotate_analytic_hill(hill_geometry)

    def _rotate_cad_hill(self, hill_geometry: trimesh.Trimesh) -> trimesh.Trimesh:
        """Rotate the CAD hill geometry about its x:sub:`2` axis.

        :param hill_geometry: The hill geometry as a
            `Trimesh <https://trimsh.org/trimesh.html>`_ object.
        :return: The rotated hill geometry as a
            `Trimesh <https://trimsh.org/trimesh.html>`_
        """
        rotation_angle_rad = (
                self.orientation * np.pi / 180
        )
        rotation_axis = np.array([0, 1, 0])

        rot_matrix = trimesh.transformations.rotation_matrix(
            rotation_angle_rad, rotation_axis
        )

        return hill_geometry.apply_transform(rot_matrix)

    def _rotate_analytic_hill(
        self, hill_geometry: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Rotate the analytic hill geometry about its x :sub:`2` axis.

        :param hill_geometry: The hill geometry as a dictionary of 1-D NumPy ndarrays
            of shape (n, ) representing the geometry's Cartesian coordinates, where n
            denotes the number of available discrete points of the corresponding 3-D
            geometry point cloud.
        :return: The rotated hill geometry as a dictionary of 1-D NumPy ndarrays of
            shape (n, ) representing the geometry's Cartesian coordinates, where n
            denotes the number of available discrete points of the corresponding 3-D
            geometry point cloud.
        """
        hill_angle_rad = self.orientation * np.pi / 180

        x_1_rot = (
            np.cos(hill_angle_rad) * hill_geometry["X"]
            + np.sin(hill_angle_rad) * hill_geometry["Z"]
        )
        x_2_rot = hill_geometry["Y"]
        x_3_rot = (
            -np.sin(hill_angle_rad) * hill_geometry["X"]
            + np.cos(hill_angle_rad) * hill_geometry["Z"]
        )

        hill_geometry["X"] = x_1_rot
        hill_geometry["Y"] = x_2_rot
        hill_geometry["Z"] = x_3_rot

        return hill_geometry

    def _compute_polynomial_coefficients(self) -> np.ndarray:
        """Compute the hill's fifth-degree polynomial coefficients.

        :return: A 1-D NumPy ndarray of shape (6, ) containing the polynomial
            coefficients.
        """
        x_vec = [self.cyl_section_width_m / 2, self.width_m / 2]
        b_vec = np.array([[self.height_m], [0], [0], [0], [0], [0]])
        a_mat = np.array(
            [
                [
                    x_vec[0] ** 5,
                    x_vec[0] ** 4,
                    x_vec[0] ** 3,
                    x_vec[0] ** 2,
                    x_vec[0],
                    1,
                ],
                [
                    5 * x_vec[0] ** 4,
                    4 * x_vec[0] ** 3,
                    3 * x_vec[0] ** 2,
                    2 * x_vec[0],
                    1,
                    0,
                ],
                [20 * x_vec[0] ** 3, 12 * x_vec[0] ** 2, 6 * x_vec[0], 2, 0, 0],
                [
                    x_vec[1] ** 5,
                    x_vec[1] ** 4,
                    x_vec[1] ** 3,
                    x_vec[1] ** 2,
                    x_vec[1],
                    1,
                ],
                [
                    5 * x_vec[1] ** 4,
                    4 * x_vec[1] ** 3,
                    3 * x_vec[1] ** 2,
                    2 * x_vec[1],
                    1,
                    0,
                ],
                [20 * x_vec[1] ** 3, 12 * x_vec[1] ** 2, 6 * x_vec[1], 2, 0, 0],
            ]
        )

        return np.linalg.solve(a_mat, b_vec).reshape((6,))

    def _compute_fifth_degree_polynomial(
        self, x_1: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Probe the fifth-degree polynomial profile of the hill.

        :param x_1: Cartesian coordinates of the probes in the x:sub:`1`-direction as a
            single real number or a 1-D NumPy ndarray of shape (n,), where n denotes
            the number of probe points.
        :return: Cartesian coordinates of the profile in the x:sub:`2`-direction at the
            specified probe points.
        """
        return (
            self.polynomial_coefficients[0] * x_1**5
            + self.polynomial_coefficients[1] * x_1**4
            + self.polynomial_coefficients[2] * x_1**3
            + self.polynomial_coefficients[3] * x_1**2
            + self.polynomial_coefficients[4] * x_1
            + self.polynomial_coefficients[5]
        )

    def _compute_corner(
        self,
        u_p: Union[float, np.ndarray],
        v_p: Union[float, np.ndarray],
        sign_x_1: Literal[-1, 1],
        sign_x_3: Literal[-1, 1],
    ) -> Tuple[
        Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
    ]:
        """Construct one three-dimensional hill corner using the analytic
        parametrization (now obsolete) in a specified quadrant.

        :param u_p: First parameter of the parametric corner function as real number or
            a 1-D NumPy ndarray of shape (n, ), where n denotes the number of discrete
            probing points.
        :param v_p: Second parameter of the parametric corner function as real number
            or a 1-D NumPy ndarray of shape (n,), where n denotes the number of
            discrete probing points.
        :param sign_x_1: Identifier for the quadrant of the hill corner in the
            x:sub:`1`-direction, specified using -1 or 1.
        :param sign_x_3: Identifier for the quadrant of the hill corner in the
            x:sub:`3`-direction, specified using -1 or 1.
        :return: A tuple of three real numbers or NumPy 1-D ndarrays or shape (n, ),
            where n denotes the number of discrete probing points, representing the
            Cartesian coordinates of the hill corner in the specified quadrant.
        """
        x_1 = (sign_x_1 * self.cyl_section_width_m / 2) + (
            u_p - self.cyl_section_width_m / 2
        ) * (abs(np.cos(v_p)) ** 0.5) * np.sign(np.cos(v_p))

        x_2 = self._compute_fifth_degree_polynomial(u_p)

        x_3 = (sign_x_3 * self.cyl_section_width_m / 2) + (
            u_p - self.cyl_section_width_m / 2
        ) * (abs(np.sin(v_p)) ** 0.5) * np.sign(np.sin(v_p))

        return x_1, x_2, x_3

    def _compute_flat_top(
        self,
        u_p: Union[float, np.ndarray],
        v_p: Union[float, np.ndarray],
    ) -> Tuple[
        Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
    ]:
        """Construct the flat top of the hill using the analytic parametrization.

        :param u_p: First parameter of the parametric function as a real number or a
            1-D NumPy ndarray of shape (n, ), where n denotes the number of discrete
            probing points.
        :param v_p: First parameter of the parametric function as a real number or a
            1-D NumPy ndarray of shape (n, ), where n denotes the number of discrete
            probing points.
        :return: A tuple of three real numbers or NumPy 1-D ndarrays or shape (n, ),
            where n denotes the number of discrete probing points, representing the
            Cartesian coordinates of the hill top.
        """
        x_1 = u_p
        x_2 = self.height_m
        x_3 = v_p
        return x_1, x_2, x_3

    def _compute_cyl_surface(
        self,
        u_p: Union[float, np.ndarray],
        v_p: Union[float, np.ndarray],
        quadrant: Literal[1, 2, 3, 4],
    ) -> Tuple[
        Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
    ]:
        """Construct the cylindrical section of the hill using the analytic
        parametrization in a specified quadrant.

        :param u_p: First parameter of the parametric function as a real number or a
            1-D NumPy ndarray of shape (n, ), where n denotes the number of discrete
            probing points.
        :param v_p: Second parameter of the parametric function as a real number or a
            1-D NumPy ndarray of shape (n, ), where n denotes the number of discrete
            probing points.
        :param quadrant: The desired quadrant, specified using 1, 2, 3, or 4.
        :return: A tuple of three real numbers or NumPy 1-D ndarrays or shape (n, ),
            where n denotes the number of discrete probing points, representing the
            Cartesian coordinates of the cylindrical section in the specified quadrant.
        """
        if quadrant == 1:
            x_1 = u_p
            x_2 = self._compute_fifth_degree_polynomial(u_p)
            x_3 = v_p
        elif quadrant == 2:
            x_1 = u_p
            x_2 = self._compute_fifth_degree_polynomial(v_p)
            x_3 = v_p
        elif quadrant == 3:
            x_1 = -u_p
            x_2 = self._compute_fifth_degree_polynomial(u_p)
            x_3 = v_p
        elif quadrant == 4:
            x_1 = u_p
            x_2 = self._compute_fifth_degree_polynomial(v_p)
            x_3 = -v_p
        else:
            raise ValueError("Invalid surface type")

        return x_1, x_2, x_3

    def _probe_analytic_hill(
        self,
        x_1_probe_m: float,
        x_3_probe_m: float,
        hill_angle_degrees: float,
    ) -> float:
        """Probe the analytic (now obsolete) hill geometry using a single probe.

        :param x_1_probe_m: The probe's x:sub:`1` coordinate, measured in meters.
        :param x_3_probe_m:  The probe's x:sub:`3` coordinate, measured in meters.
        :param hill_angle_degrees: The yaw angle orientation of the hill, measured
            in degrees.
        :return: The height of the Hill at the probe's location, measured in meters.
        """
        if not all(isinstance(x, (int, float)) for x in (x_1_probe_m, x_3_probe_m)):
            raise TypeError("x_1_probe_m and x_3_probe_m must be of type int or float")

        # Rotate coordinate system counterclockwise to align with the 0 deg hill orientation
        angle_rad = hill_angle_degrees * np.pi / 180
        x_1 = x_1_probe_m * np.cos(angle_rad) - x_3_probe_m * np.sin(angle_rad)
        x_2 = None
        x_3 = x_1_probe_m * np.sin(angle_rad) + x_3_probe_m * np.cos(angle_rad)

        # Compute hill height
        def corner_sys_of_eqns(x_var, sign_x_1, sign_x_3):
            x1_corner, _, x3_corner = self._compute_corner(
                x_var[0], x_var[1], sign_x_1, sign_x_3
            )
            return [x_1 - x1_corner, x_3 - x3_corner]

        def solve_corner(sys_of_eqns, init_guess, sign_x_1, sign_x_3):
            maxiter = 6
            angle_increment_rad = 15 * np.pi / 180
            xx_2 = None
            for i in range(maxiter):
                solution = spoptimize.fsolve(
                    sys_of_eqns, init_guess, args=(sign_x_1, sign_x_3)
                )
                xx_2 = self._compute_fifth_degree_polynomial(solution[0])
                if 0 <= xx_2 <= self.height_m:
                    break
                angle_adjustment = (-1) ** i * (1 + i // 2) * angle_increment_rad
                init_guess = np.array([init_guess[0], init_guess[1] + angle_adjustment])
            return xx_2

        if (x_1 > self.cyl_section_width_m / 2) and (
            x_3 > self.cyl_section_width_m / 2
        ):
            x_2 = solve_corner(corner_sys_of_eqns, np.array([x_1, np.pi / 4]), 1, 1)
        elif (x_1 < -self.cyl_section_width_m / 2) and (
            x_3 > self.cyl_section_width_m / 2
        ):
            x_2 = solve_corner(
                corner_sys_of_eqns, np.array([-x_1, 3 * np.pi / 4]), -1, 1
            )
        elif (x_1 < -self.cyl_section_width_m / 2) and (
            x_3 < -self.cyl_section_width_m / 2
        ):
            x_2 = solve_corner(
                corner_sys_of_eqns, np.array([-x_1, 5 * np.pi / 4]), -1, -1
            )
        elif (x_1 > self.cyl_section_width_m / 2) and (
            x_3 < -self.cyl_section_width_m / 2
        ):
            x_2 = solve_corner(
                corner_sys_of_eqns, np.array([x_1, 7 * np.pi / 4]), 1, -1
            )
        elif (x_1 > self.cyl_section_width_m / 2) and (
            -self.cyl_section_width_m / 2 <= x_3 <= self.cyl_section_width_m / 2
        ):
            x_2 = self._compute_fifth_degree_polynomial(x_1)
        elif (x_1 < -self.cyl_section_width_m / 2) and (
            -self.cyl_section_width_m / 2 <= x_3 <= self.cyl_section_width_m / 2
        ):
            x_2 = self._compute_fifth_degree_polynomial(-x_1)
        elif (x_3 < -self.cyl_section_width_m / 2) and (
            -self.cyl_section_width_m / 2 <= x_1 <= self.cyl_section_width_m / 2
        ):
            x_2 = self._compute_fifth_degree_polynomial(-x_3)
        elif (x_3 > self.cyl_section_width_m / 2) and (
            -self.cyl_section_width_m / 2 <= x_1 <= self.cyl_section_width_m / 2
        ):
            x_2 = self._compute_fifth_degree_polynomial(x_3)
        elif (-self.cyl_section_width_m <= x_1 <= self.cyl_section_width_m / 2) and (
            -self.cyl_section_width_m <= x_3 <= self.cyl_section_width_m / 2
        ):
            x_2 = self.height_m

        return x_2

    def _probe_cad_hill(self, x_1_m: float, x_3_m: float) -> float:
        """Probe the BeVERLI Hill CAD geometry at a single location using ray tracing.

        :param x_1_m: The probe's x:sub:`1` coordinate, measured in meters.
        :param x_3_m: The probe's x:sub:`3` coordinate, measured in meters.
        :return: The height of the Hill at the probe's location, measured in meters.
        """
        if not all(isinstance(x, (int, float)) for x in (x_1_m, x_3_m)):
            raise TypeError("x_1_probe_m and x_3_probe_m must be of type int or float")

        # Define a ray with starting point and direction
        ray_origin = [x_1_m, -0.1, x_3_m]
        ray_direction = [0, 1, 0]

        intersections, _, _ = self.geometry.ray.intersects_location(
            ray_origins=[ray_origin], ray_directions=[ray_direction]
        )

        try:
            if len(intersections) == 0:
                raise ValueError("No intersection found with the mesh.\n")
        except ValueError as err:
            print(f"{err}")
            response = input("Do you wish to proceed with zero height? (y/n) ")
            while response.lower() not in ['y', 'n']:
                response = input("Do you wish to proceed with zero height? (y/n) ")
            if response == 'n':
                sys.exit(1)
            else:
                return 0

        max_height_intersection = max(intersections, key=lambda point: point[1])
        height = max_height_intersection[1]

        return height

    def _compute_corners_perimeter(
        self, num_of_pts: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute the superelliptic corners of the hill perimeter.

        :param num_of_pts: An integer representing the number of discrete points used
            to discretize each corner.
        :return: A list of tuples of three 1-D NumPy ndarrays of shape (n, ), where n
            denotes the number of discrete points, each array representing a Cartesian
            coordinate of a specific elliptic corner.
        """
        corners = []
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 4, endpoint=False)):
            x_1_corner, _, x_3_corner = self._compute_corner(
                self.width_m / 2,
                np.linspace(angle, angle + np.pi / 2, num_of_pts),
                (-1) ** (i + (i + 1) // 3),
                (-1) ** (i // 2),
            )
            corners.append((x_1_corner, _, x_3_corner))
        return corners

    def _calculate_edges_perimeter(
        self, num_of_pts: int
    ) -> List[Tuple[np.ndarray, int, np.ndarray]]:
        """Compute the straight edges of the hill perimeter.

        :param num_of_pts: Number of points to discretize each straight edge of the
            hill perimeter.
        :return: A list of tuples of three 1-D NumPy ndarrays of shape (n,), where n
            denotes the number of discrete points, each array representing a Cartesian
            coordinate of a specific edge.
        """
        edges = []
        sign = -1
        for i in range(4):
            if i % 2 == 0:
                sign = -sign
                x_1_edge = sign * self.width_m / 2 * np.ones((num_of_pts - 2,))
                x_3_edge = np.linspace(
                    -sign * self.cyl_section_width_m / 2,
                    sign * self.cyl_section_width_m / 2,
                    num_of_pts,
                )[1:-1]
            else:
                x_1_edge = np.linspace(
                    sign * self.cyl_section_width_m / 2,
                    -sign * self.cyl_section_width_m / 2,
                    num_of_pts,
                )[1:-1]
                x_3_edge = sign * self.width_m / 2 * np.ones((num_of_pts - 2,))
            edges.append((x_1_edge, 0, x_3_edge))
        return edges

    @staticmethod
    def _find_perimeter_intersection_points(
        x_1: np.ndarray,
        x_3: np.ndarray,
        x_3_intersect: float,
    ) -> np.ndarray:
        """Find intersection points of hill's perimeter with the
        x:sub:`1`-x:sub:`2`-plane at a desired probe point.

        :param x_1: A 1-D NumPy ndarray of shape (n,), where n denotes the number of
            discretization points, representing the hill perimeter's
            x:sub:`1` coordinate.
        :param x_3: A 1-D NumPy ndarray of shape (n,), where n denotes the number of
            discretization points, representing the hill perimeter's
            x:sub:`3` coordinate.
        :param x_3_intersect: Desired x:sub:`3` location of the probing plane.
        :return: A 1-D NumPy ndarray of shape (2, ) containing the two intersection
            points of the probing plane with the hill perimeter.
        """
        # Find crossing points
        sign_x_3 = np.sign(x_3 - x_3_intersect)
        diff_sign_x_3 = np.diff(sign_x_3)
        intersection_indices = np.where(diff_sign_x_3 != 0)
        # Check if the first and last element build a crossing point
        if (sign_x_3[0] - sign_x_3[-1]) != 0:
            intersection_indices = np.append(intersection_indices, len(x_3) - 1, 0)
        intersection_indices = np.unique(intersection_indices)
        # Check also that only 2 intersection points exist
        if len(intersection_indices) > 2:
            raise ValueError("There should only be only 2 crossing points!")

        x_1_intersect = np.zeros((2,))
        x_1_intersect[0] = np.interp(
            0,
            np.array([x_3[intersection_indices[0]], x_3[intersection_indices[0] + 1]]),
            np.array([x_1[intersection_indices[0]], x_1[intersection_indices[0] + 1]]),
        )
        x_1_intersect[1] = np.interp(
            0,
            np.array([x_3[intersection_indices[1]], x_3[intersection_indices[1] + 1]]),
            np.array([x_1[intersection_indices[1]], x_1[intersection_indices[1] + 1]]),
        )

        return x_1_intersect
