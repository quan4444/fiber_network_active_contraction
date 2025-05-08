import meshio
import numpy as np
from dataclasses import dataclass,field
from scipy.stats import vonmises
from shapely.geometry import LineString, MultiLineString,Polygon,Point,GeometryCollection
from shapely.ops import unary_union
from shapely.strtree import STRtree
from typing import Optional,Union,List,Dict


def hello_world():
    print("Hello, World! This is fiber_network package.")


def generate_sample_tissue_params() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Generate parameters for a sample synthetic microtissue.

    Returns:
        tuple: A tuple containing:
            - tissue_pts (np.ndarray): Array of shape (N, 2) representing the tissue boundary points.
            - posts_pos (np.ndarray): Array of shape (M, 2) representing the positions of microposts.
            - posts_radius (float): Radius of the microposts.
            - wound_shape (np.ndarray): Array of shape (P, 2) representing the wound shape as a polygon.
    """
    # Define tissue boundary points
    tissue_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.678899083],
        [0.0, 0.678899083]
    ])

    # Define micropost positions and apply a small shift
    posts_pos = np.array([
        [0.100917431, 0.110091743],
        [0.899082569, 0.110091743],
        [0.100917431, 0.581039755],
        [0.899082569, 0.577981651]
    ])
    post_shift = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]]) * 0.05
    posts_pos += post_shift

    # Define micropost radius
    posts_radius = 0.1

    # Define wound shape as an ellipse
    center = Point(0.5, 0.34480122)
    semi_major_axis = 0.05
    semi_minor_axis = 0.05
    num_points = 100  # Number of points to approximate the ellipse
    theta_values = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    wound_shape = np.array([
        [
            center.x + semi_major_axis * np.cos(theta),
            center.y + semi_minor_axis * np.sin(theta)
        ]
        for theta in theta_values
    ])

    return tissue_pts, posts_pos, posts_radius, wound_shape


@dataclass
class FiberNetworkParams:
    """
    Unified configuration for generating a fiber network.
    """
    # Tissue parameters
    tissue_points: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    dimension: int = 2

    # Fiber parameters
    num_fibers: int = 100
    fiber_length: Dict[str, Union[str, float]] = field(default_factory=lambda: {
        "distribution": "uniform",  # Options: "constant", "uniform", "normal"
        "min": 0.1,
        "max": 0.5,
        "mean": 0.3,
        "std_dev": 0.05
    })
    fiber_orientation: Dict[str, Union[str, float, List[float]]] = field(default_factory=lambda: {
        "distribution": "von_mises",  # Options: "random", "uniform", "von_mises"
        "range": [0, 180],  # For "uniform" distribution
        "mean_angle": 0,  # For "von_mises" distribution
        "concentration": 1.0  # For "von_mises" distribution
    })
    fiber_radii: Dict[str, Union[str, float]] = field(default_factory=lambda: {
        "distribution": "normal",  # Options: "constant", "uniform", "normal"
        "mean": 0.01,
        "std_dev": 0.002
    })

    # Micropost parameters
    post_positions: np.ndarray = field(default_factory=lambda: np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]]))
    post_radius: float = 0.1
    post_tolerance: float = 1e-14

    # Wound parameters
    wound_shape: Optional[np.ndarray] = field(default_factory=lambda: np.array([[0.4,0.4],[0.6,0.4],[0.6,0.6],[0.4,0.6]]))

    # Patching parameters
    num_patches_x: int = 1
    num_patches_y: int = 1

    def validate(self):
        """
        Validate the configuration parameters.
        """
        if self.tissue_points.ndim != 2 or self.tissue_points.shape[1] != self.dimension:
            raise ValueError(f"tissue_points must be a 2D array with shape (N, {self.dimension}).")
        if self.num_fibers <= 0:
            raise ValueError("num_fibers must be a positive integer.")
        if self.post_radius <= 0:
            raise ValueError("post_radius must be a positive number.")
        if self.num_patches_x <= 0 or self.num_patches_y <= 0:
            raise ValueError("num_patches_x and num_patches_y must be positive integers.")


@dataclass
class FiberNetworkResult:
    fibers: np.ndarray
    indices: np.ndarray
    radii: np.ndarray


def get_filename(
    basename: str,
    fib_n: int = 0,
    config: FiberNetworkParams = FiberNetworkParams(),
    slit_option: bool = False,
    slit_size: Union[float, int] = 0,
    seed_num: Optional[int] = None,
) -> str:
    """
    Generate a filename based on the provided parameters.
    """
    f_name = f"{fib_n}{basename}"
    f_name += f"_len_{config.fiber_length['distribution']}"
    if config.fiber_length['distribution'] == 'uniform':
        f_name += f"_min{config.fiber_length['min']}_max{config.fiber_length['max']}"
    elif config.fiber_length['distribution'] == 'normal':
        f_name += f"_mean{config.fiber_length['mean']}_std{config.fiber_length['std_dev']}"

    f_name += f"_orien_{config.fiber_orientation['distribution']}"
    if config.fiber_orientation['distribution'] == 'uniform':
        f_name += f"_range{config.fiber_orientation['range']}"
    elif config.fiber_orientation['distribution'] == 'von_mises':
        f_name += f"_mu{config.fiber_orientation['mean_angle']}_kappa{config.fiber_orientation['concentration']}"

    f_name += f"_radii_{config.fiber_radii['distribution']}"
    if config.fiber_radii['distribution'] == 'normal':
        f_name += f"_mean{config.fiber_radii['mean']}_std{config.fiber_radii['std_dev']}"

    if slit_option:
        f_name += f"_slit_{slit_size}"
    if seed_num is not None:
        f_name += f"_seed{seed_num}"
    return f_name


def generate_random_points_in_area(shape_pts: np.ndarray, fib_n: int) -> np.ndarray:
    """
    Generate random points within a defined 2D area.

    This function generates the first points of a set of fibers (lines) within the bounding box
    defined by the given shape points.

    Parameters:
    ----------
    shape_pts : np.ndarray
        A 2D array of shape (N, 2) representing the vertices of the area (e.g., a polygon).
        Each row corresponds to a point [x, y].
    fib_n : int
        The number of random points (fibers) to generate.

    Returns:
    -------
    np.ndarray
        A 2D array of shape (fib_n, 2), where each row represents a random point [x, y]
        within the bounding box of the given shape.
    """
    # Validate inputs
    if shape_pts.ndim != 2 or shape_pts.shape[1] != 2 or shape_pts.shape[0] < 3:
        raise ValueError("shape_pts must be a 2D array with shape (N, 2), where N>=3.")
    if fib_n <= 0:
        raise ValueError("fib_n must be a positive integer.")

    # Calculate the bounding box of the shape
    min_x, min_y = np.amin(shape_pts, axis=0)
    max_x, max_y = np.amax(shape_pts, axis=0)

    # Generate random points within the bounding box
    random_points = np.random.rand(fib_n, 2) * [max_x - min_x, max_y - min_y] + [min_x, min_y]

    return random_points


def generate_points_by_patches(
    tissue_pts: np.ndarray,
    fib_n: int,
    num_patch_x: int,
    num_patch_y: int
) -> np.ndarray:
    """
    Divide the region of interest (ROI) into patches and generate an equal number of random points in each patch.

    Args:
    tissue_pts : np.ndarray
        A 2D array of shape (N, 2) representing the vertices of the ROI (e.g., a polygon).
        Each row corresponds to a point [x, y].
    fib_n : int
        The total number of random points (fibers) to generate.
    num_patch_x : int
        The number of patches along the x-axis.
    num_patch_y : int
        The number of patches along the y-axis.

    Returns:
    np.ndarray
        A 2D array of shape (fib_n, 2), where each row represents a random point [x, y]
        within the ROI divided into patches.
    """
    # Validate inputs
    if tissue_pts.ndim != 2 or tissue_pts.shape[1] != 2 or tissue_pts.shape[0] < 3:
        raise ValueError("tissue_pts must be a 2D array with shape (N, 2), where N>=3.")
    if fib_n <= 0:
        raise ValueError("fib_n must be a positive integer.")
    if num_patch_x <= 0 or num_patch_y <= 0:
        raise ValueError("num_patch_x and num_patch_y must be positive integers.")

    # Calculate the bounding box of the tissue
    min_x, min_y = np.amin(tissue_pts, axis=0)
    max_x, max_y = np.amax(tissue_pts, axis=0)

    # Calculate patch dimensions
    patch_x_len = (max_x - min_x) / num_patch_x
    patch_y_len = (max_y - min_y) / num_patch_y

    # Calculate the number of points per patch
    total_patches = num_patch_x * num_patch_y
    points_per_patch = int(np.ceil(fib_n / total_patches))

    # Generate random points for each patch
    patch_points_list = []
    for i in range(num_patch_x):
        for j in range(num_patch_y):
            # Define the patch boundaries
            patch_min_x = min_x + i * patch_x_len
            patch_max_x = patch_min_x + patch_x_len
            patch_min_y = min_y + j * patch_y_len
            patch_max_y = patch_min_y + patch_y_len

            # Generate random points within the patch
            random_points = np.random.rand(points_per_patch, 2) * [
                patch_max_x - patch_min_x,
                patch_max_y - patch_min_y,
            ] + [patch_min_x, patch_min_y]

            patch_points_list.append(random_points)

    # Combine all points and trim to the required number of points
    all_points = np.vstack(patch_points_list)[:fib_n]

    return all_points


def generate_fiber_lengths(fib_n: int, len_dist_type: str, min_fib_L: float, max_fib_L: float, 
                           mean_fib_L: float, sd_fib_L: float) -> np.ndarray:
    """Generate fiber lengths based on the specified distribution."""
    if len_dist_type == 'constant':
        return np.full((fib_n, 1), (max_fib_L + min_fib_L) / 2)
    elif len_dist_type == 'uniform':
        return np.random.uniform(min_fib_L, max_fib_L, size=(fib_n, 1))
    elif len_dist_type == 'normal':
        return np.random.normal(mean_fib_L, sd_fib_L, size=(fib_n, 1))
    else:
        raise ValueError(f"Invalid length distribution type: {len_dist_type}")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize a set of vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms==0]=1
    return vectors / norms


def randomize_directions(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Randomize the direction of vectors."""
    signs = np.random.choice([-1, 1], size=(vectors.shape[0], dim))
    return vectors * signs


def generate_random_angles(fib_n: int, ang_dist_type: str, ang_range: Union[List, np.ndarray], 
                           ang_mu: float, ang_kappa: float, dim: int) -> np.ndarray:
    """Generate random angles based on the specified distribution."""
    if ang_dist_type == 'random_orientation':
        unit_vecs = np.random.uniform(-1, 1, size=(fib_n, dim))
        return normalize_vectors(unit_vecs)
    elif ang_dist_type == 'uniform':
        min_angle, max_angle = np.radians(ang_range)
        random_angles = np.random.uniform(min_angle, max_angle, size=fib_n)
        angle_vecs = np.column_stack((np.cos(random_angles), np.sin(random_angles)))
        return normalize_vectors(angle_vecs)
    elif ang_dist_type == 'von_mises':
        if ang_kappa <= 0:
            ang_kappa = 1e-14  # Small positive value to avoid errors
        random_angles = vonmises(loc=ang_mu, kappa=ang_kappa).rvs(size=fib_n)
        angle_vecs = np.column_stack((np.cos(random_angles), np.sin(random_angles)))
        return normalize_vectors(angle_vecs)
    else:
        raise ValueError(f"Invalid angle distribution type: {ang_dist_type}")


def generate_fib_second_pts(
    first_pts: np.ndarray,
    min_fib_L: Union[float, int],
    max_fib_L: Union[float, int],
    mean_fib_L: Union[float, int],
    sd_fib_L: Union[float, int],
    ang_range: Union[List, bool, np.ndarray],
    ang_mu: Union[float, int],
    ang_kappa: Union[float, int],
    len_dist_type: str = 'uniform',
    ang_dist_type: str = 'von_mises',
    dim: int = 2,
) -> np.ndarray:
    """
    Generate the second points for multiple fibers by randomizing a vector and adding it to the first points.
    """
    # Validate inputs
    if first_pts.ndim != 2 or first_pts.shape[1] != dim:
        raise ValueError(f"first_pts must be a 2D array with shape (N, {dim}).")

    fib_n = first_pts.shape[0]

    # Generate fiber lengths and angles
    length_of_fibers = generate_fiber_lengths(fib_n, len_dist_type, min_fib_L, max_fib_L, mean_fib_L, sd_fib_L)
    angle_vecs = generate_random_angles(fib_n, ang_dist_type, ang_range, ang_mu, ang_kappa, dim)

    # Randomize directions and calculate second points
    angle_vecs = randomize_directions(angle_vecs, dim)
    second_pts = first_pts + length_of_fibers * angle_vecs

    return second_pts


def generate_initial_fibers(
    tissue_pts: np.ndarray,
    fib_n: int,
    min_fib_L: Union[float, int],
    max_fib_L: Union[float, int],
    mean_fib_L: Union[float, int],
    sd_fib_L: Union[float, int],
    ang_range: Union[List, bool, np.ndarray],
    ang_mu: Union[float, int],
    ang_kappa: Union[float, int],
    len_dist_type: str = 'uniform',
    ang_dist_type: str = 'von_mises',
    num_patch_x: int = 1,
    num_patch_y: int = 1,
    dim: int = 2,
) -> np.ndarray:
    """
    Generate pairs of points depicting fibers with random orientations.

    Args:
        tissue_pts (np.ndarray): Points outlining the tissue.
        fib_n (int): Number of initial fibers.
        min_fib_L (float): Minimum length of a fiber.
        max_fib_L (float): Maximum length of a fiber.
        mean_fib_L (float): Mean length of a fiber (for normal distribution).
        sd_fib_L (float): Standard deviation of fiber length (for normal distribution).
        ang_range (Union[List, bool, np.ndarray]): Range of angles for uniform distribution.
        ang_mu (float): Mean angle for von Mises distribution.
        ang_kappa (float): Concentration parameter for von Mises distribution.
        len_dist_type (str): Type of length distribution ('uniform', 'normal', 'constant').
        ang_dist_type (str): Type of angle distribution ('von_mises', 'uniform', 'random_orientation').
        num_patch_x (int): Number of patches along the x-axis.
        num_patch_y (int): Number of patches along the y-axis.
        dim (int): Dimension of the tissue (2D or 3D).

    Returns:
        np.ndarray: Array of shape (fib_n, 2 * dim) containing pairs of points for each fiber.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # Validate inputs
    if tissue_pts.ndim != 2 or tissue_pts.shape[1] != dim:
        raise ValueError(f"tissue_pts must be a 2D array with shape (N, {dim}).")
    if fib_n <= 0:
        raise ValueError("fib_n must be a positive integer.")
    if num_patch_x <= 0 or num_patch_y <= 0:
        raise ValueError("num_patch_x and num_patch_y must be positive integers.")
    if len_dist_type not in ['constant', 'uniform', 'normal']:
        raise ValueError(f"Invalid len_dist_type: {len_dist_type}. Must be 'constant', 'uniform', or 'normal'.")
    if ang_dist_type not in ['random_orientation', 'uniform', 'von_mises']:
        raise ValueError(f"Invalid ang_dist_type: {ang_dist_type}. Must be 'random_orientation', 'uniform', or 'von_mises'.")

    # Generate the first points within the tissue
    first_pts = generate_points_by_patches(
        tissue_pts=tissue_pts,
        fib_n=fib_n,
        num_patch_x=num_patch_x,
        num_patch_y=num_patch_y,
    )

    # Generate the second points based on orientations and lengths parameters
    last_pts = generate_fib_second_pts(
        first_pts=first_pts,
        min_fib_L=min_fib_L,
        max_fib_L=max_fib_L,
        mean_fib_L=mean_fib_L,
        sd_fib_L=sd_fib_L,
        ang_range=ang_range,
        ang_mu=ang_mu,
        ang_kappa=ang_kappa,
        len_dist_type=len_dist_type,
        ang_dist_type=ang_dist_type,
        dim=dim,
    )

    # Concatenate the first and second points to form fibers
    all_pts = np.concatenate((first_pts, last_pts), axis=1)

    return all_pts


def cut_network_into_shape(
    lines: np.ndarray, 
    shape_vertices: np.ndarray
) -> np.ndarray:
    """
    Cut a fiber network (lines) to fit within a given shape (shape_vertices).
    If a line is partially outside the shape, only the portion inside the shape is retained.

    Args:
        lines (np.ndarray): Array of shape (N, 4) representing pairs of points depicting lines (fibers).
                            Each row is [x1, y1, x2, y2].
        shape_vertices (np.ndarray): Array of shape (M, 2) representing the vertices of the shape.

    Returns:
        np.ndarray: Array of shape (K, 4) representing the lines that fit within the shape.
    """
    # Validate inputs
    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError("lines must be a 2D array with shape (N, 4).")
    if shape_vertices.ndim != 2 or shape_vertices.shape[1] != 2 or shape_vertices.shape[0] < 3:
        raise ValueError("shape_vertices must be a 2D array with shape (N, 2), where N>=3.")

    # Create a polygon representing the shape
    shape_polygon = Polygon(shape_vertices)
    if not shape_polygon.is_valid:
        raise ValueError("shape_vertices do not form a valid polygon.")

    # Convert lines to LineString objects
    line_strings = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lines]

    # Process each line to check its relationship with the shape
    cut_lines = []
    for line in line_strings:
        # Check if both endpoints are inside the shape
        if shape_polygon.contains(Point(line.coords[0])) and shape_polygon.contains(Point(line.coords[-1])):
            coords = list(line.coords)
            cut_lines.append([coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]])
            continue

        # Find the intersection between the line and the shape
        intersection = line.intersection(shape_polygon)
        if not intersection.is_empty:
            if isinstance(intersection, LineString):
                # If the intersection is a single line, add it to the result
                coords = list(intersection.coords)
                cut_lines.append([coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]])
            elif isinstance(intersection, MultiLineString):
                # If the intersection is multiple lines, add each segment
                for segment in intersection.geoms:
                    coords = list(segment.coords)
                    cut_lines.append([coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]])

    # Convert the result to a NumPy array
    if cut_lines:
        cut_lines_array = np.array(cut_lines)
    else:
        cut_lines_array = np.empty((0, 4))  # Return an empty array if no lines remain

    return cut_lines_array


def remove_lines_in_circles(
    circle_centers: np.ndarray,
    circle_radius: Union[float, int],
    lines: np.ndarray
) -> np.ndarray:
    """
    Remove all lines inside given circles. 
    If a line is both inside and outside, retain only the portion outside.

    Args:
        circle_centers (np.ndarray): Array of shape (N, 2) representing the 2D coordinates of circle centers.
        circle_radius (Union[float, int]): Radius of all circles.
        lines (np.ndarray): Array of shape (M, 4) representing pairs of points depicting lines (fibers).
                            Each row is [x1, y1, x2, y2].

    Returns:
        np.ndarray: Array of shape (K, 4) representing the lines after removal.
    """
    # Validate inputs
    if circle_centers.ndim != 2 or circle_centers.shape[1] != 2:
        raise ValueError("circle_centers must be a 2D array with shape (N, 2).")
    if not isinstance(circle_radius, (float, int)) or circle_radius <= 0:
        raise ValueError("circle_radius must be a positive number.")
    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError("lines must be a 2D array with shape (M, 4).")

    # Create Shapely geometries for circles and lines
    circle_geometries = [Point(center).buffer(circle_radius) for center in circle_centers]
    line_strings = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lines]

    # Process each line to check for intersections with circles
    cut_lines = []
    for line in line_strings:
        line_cut = line
        for circle in circle_geometries:
            # Subtract the circle from the line
            line_cut = line_cut.difference(circle)
            if line_cut.is_empty:
                break  # If the line is completely inside a circle, skip further processing
        cut_lines.append(line_cut)

    # Convert the resulting geometries back to a NumPy array
    resulting_lines_array = convert_linestring_multilinestring_to_nparray(cut_lines)

    # Sort and remove duplicate lines
    sorted_lines = sort_points_and_lines(resulting_lines_array)

    return sorted_lines


def convert_linestring_multilinestring_to_nparray(geometries: List) -> np.ndarray:
    """
    Convert a list of Shapely LineString or MultiLineString objects into a NumPy array.

    Args:
        geometries (List): A list of Shapely LineString, MultiLineString, or GeometryCollection objects.

    Returns:
        np.ndarray: A 2D array of shape (N, 4), where each row represents a line [x1, y1, x2, y2].
    """
    # Validate input
    if not isinstance(geometries, list):
        geometries = [geometries]  # Wrap single geometry in a list

    # Initialize a list to store line coordinates
    coordinates = []

    for geometry in geometries:
        if isinstance(geometry, LineString):
            # Extract coordinates from LineString
            if not geometry.is_empty:
                coords = list(geometry.coords)
                if len(coords) == 2:  # Ensure it's a valid line
                    coordinates.append([coords[0][0], coords[0][1], coords[1][0], coords[1][1]])

        elif isinstance(geometry, MultiLineString):
            # Extract coordinates from each LineString in MultiLineString
            for line in geometry.geoms:
                if not line.is_empty:
                    coords = list(line.coords)
                    if len(coords) == 2:  # Ensure it's a valid line
                        coordinates.append([coords[0][0], coords[0][1], coords[1][0], coords[1][1]])

        elif isinstance(geometry, GeometryCollection):
            # Extract coordinates from each geometry in GeometryCollection
            for geom in geometry.geoms:
                if isinstance(geom, LineString) and not geom.is_empty:
                    coords = list(geom.coords)
                    if len(coords) == 2:  # Ensure it's a valid line
                        coordinates.append([coords[0][0], coords[0][1], coords[1][0], coords[1][1]])

    # Convert the list of coordinates to a NumPy array and remove duplicates
    if coordinates:
        coordinates_array = np.unique(np.array(coordinates), axis=0)
    else:
        coordinates_array = np.empty((0, 4))  # Return an empty array if no valid lines are found

    return coordinates_array


def remove_non_intersecting_fibers(lines:np.ndarray) -> np.ndarray:
    """
    Remove lines that are not in contact with any other lines.

    Args:
        lines (np.ndarray): Array of shape (N, 4) representing pairs of points depicting lines (fibers).
                            Each row is [x1, y1, x2, y2].

    Returns:
        np.ndarray: Array of shape (K, 4) containing only lines that intersect with at least one other line.
    """

    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError("lines must be a 2D array with shape (N, 4).")

    intersecting_fibers = []
    for i in range(len(lines)):
        x1,y1,x2,y2= lines[i,:]
        line1 = LineString([(x1, y1), (x2, y2)])
        is_intersecting = False

        for j in range(len(lines)):
            if i != j:
                x3,y3,x4,y4 = lines[j,:]
                line2 = LineString([(x3, y3), (x4, y4)])

                if line1.intersects(line2):
                    is_intersecting = True
                    break

        if is_intersecting: # if line1 intersects with any other line, append to new list
            intersecting_fibers.append(lines[i])
    intersecting_fibers = np.array(intersecting_fibers)
    return intersecting_fibers


def remove_lines_in_wound(lines: np.ndarray, shape_vertices: np.ndarray) -> np.ndarray:
    """
    Remove all lines inside a given polygon (wound shape).
    If a line is partially inside and partially outside, retain only the portion outside.

    Args:
        lines (np.ndarray): Array of shape (N, 4) representing pairs of points depicting lines (fibers).
                            Each row is [x1, y1, x2, y2].
        shape_vertices (np.ndarray): Array of shape (M, 2) representing the vertices of the polygon.

    Returns:
        np.ndarray: Array of shape (K, 4) representing the lines after removal.
    """
    # Validate inputs
    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError("lines must be a 2D array with shape (N, 4).")
    if shape_vertices.ndim != 2 or shape_vertices.shape[1] != 2 or shape_vertices.shape[0] < 3:
        raise ValueError("shape_vertices must be a 2D array with shape (M, 2), where M >= 3.")

    # Define the wound geometry as a Shapely polygon
    wound_polygon = Polygon(shape_vertices)
    if not wound_polygon.is_valid:
        raise ValueError("shape_vertices do not form a valid polygon.")

    # Convert lines to Shapely LineString objects
    line_strings = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lines]

    # Process each line to check for intersections with the wound polygon
    cut_lines = []
    for line in line_strings:
        # Subtract the wound polygon from the line
        line_cut = line.difference(wound_polygon)

        # If the result is a LineString, add it to the result
        if isinstance(line_cut, LineString) and not line_cut.is_empty:
            coords = list(line_cut.coords)
            cut_lines.append([coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]])

        # If the result is a MultiLineString, add each segment
        elif isinstance(line_cut, MultiLineString):
            for segment in line_cut.geoms:
                if not segment.is_empty:
                    coords = list(segment.coords)
                    cut_lines.append([coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]])

    # Convert the resulting lines to a NumPy array
    if cut_lines:
        resulting_lines_array = np.array(cut_lines)
    else:
        resulting_lines_array = np.empty((0, 4))  # Return an empty array if no lines remain

    # Sort and remove duplicate lines
    sorted_lines = sort_points_and_lines(resulting_lines_array)

    return sorted_lines


def sort_points_and_lines(lines: np.ndarray) -> np.ndarray:
    """
    Sort the endpoints of lines so that the smaller indexing point comes first,
    and remove duplicate lines.

    Args:
        lines (np.ndarray): Array of shape (N, 4), where each row represents a line [x1, y1, x2, y2].

    Returns:
        np.ndarray: Array of shape (M, 4), where M <= N, with sorted and unique lines.

    Example:
        >>> lines = np.array([[0.5, 0.5, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]])
        >>> sort_points_and_lines(lines)
        array([[0, 1, 1, 0],
               [0.5, 0.5, 1, 0]])
    """
    # Validate input
    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError("lines must be a 2D array with shape (N, 4).")

    lines_sorted_points = []
    for i in range(lines.shape[0]):
        line = lines[i]
        px = [line[0],line[2]]
        py = [line[1],line[3]]

        # Sort points by y coordinate, then by x coordinate
        ind_ = np.lexsort((py,px))
        line_sorted =[px[ind_[0]],py[ind_[0]],px[ind_[1]],py[ind_[1]]]
        lines_sorted_points.append(line_sorted)
    lines_sorted_points = np.unique(lines_sorted_points,axis=0)
    return np.array(lines_sorted_points)


def split_large_fibers_by_intersections(lines: np.ndarray, tolerance: float = 0.2) -> np.ndarray:
    """
    Split large fibers into smaller segments at their intersections.

    Args:
        lines (np.ndarray): Array of shape (N, 4) representing pairs of points depicting lines (fibers).
                            Each row is [x1, y1, x2, y2].
        tolerance (float): Tolerance for simplifying the MultiLineString to remove overlapping segments.

    Returns:
        np.ndarray: Array of shape (M, 4), where M >= N, representing the split fibers.
    """
    # Validate input
    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError("lines must be a 2D array with shape (N, 4).")
    if tolerance <= 0:
        raise ValueError("tolerance must be a positive number.")

    # Convert lines to Shapely LineString objects
    line_strings = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lines]

    # Combine the LineString objects into a single MultiLineString
    multi_line = unary_union(line_strings)

    # Simplify the MultiLineString to remove overlapping segments
    simplified_multi_line = multi_line.simplify(tolerance, preserve_topology=True)

    # Convert the simplified MultiLineString back to a NumPy array
    simplified_lines = convert_linestring_multilinestring_to_nparray(simplified_multi_line)

    # Sort and remove duplicate lines
    sorted_lines = sort_points_and_lines(simplified_lines)

    return sorted_lines


def create_slit_wound(
    slit: np.ndarray,
    slit_dir: str = 'x',
    slit_dist_from_mid: Union[float, int] = 0.001
) -> np.ndarray:
    """
    Create a rectangular slit wound shape based on the given slit direction and distance from the midpoint.

    Args:
        slit (np.ndarray): Array of shape (2, 2) representing the start and end points of the slit.
                           Each row is [x, y].
        slit_dir (str): Direction of the slit ('x' or 'y').
        slit_dist_from_mid (Union[float, int]): Distance from the midpoint of the slit to define the slit width.

    Returns:
        np.ndarray: Array of shape (4, 2) representing the vertices of the slit wound polygon.
    """
    # Validate inputs
    if slit.shape != (2, 2):
        raise ValueError("slit must be a 2D array with shape (2, 2).")
    if slit_dir not in ['x', 'y']:
        raise ValueError("slit_dir must be either 'x' or 'y'.")
    if slit_dist_from_mid <= 0:
        raise ValueError("slit_dist_from_mid must be a positive number.")

    # Calculate the four vertices of the slit wound
    if slit_dir == 'x':
        # Horizontal slit
        slit_p1 = slit[0] + [-slit_dist_from_mid, 0]
        slit_p2 = slit[0] + [slit_dist_from_mid, 0]
        slit_p3 = slit[1] + [slit_dist_from_mid, 0]
        slit_p4 = slit[1] + [-slit_dist_from_mid, 0]
    elif slit_dir == 'y':
        # Vertical slit
        slit_p1 = slit[0] + [0, -slit_dist_from_mid]
        slit_p2 = slit[0] + [0, slit_dist_from_mid]
        slit_p3 = slit[1] + [0, slit_dist_from_mid]
        slit_p4 = slit[1] + [0, -slit_dist_from_mid]

    # Combine the vertices into a single array
    slit_shape = np.vstack([slit_p1, slit_p2, slit_p3, slit_p4])

    return slit_shape


def lines_on_posts(
    lines: np.ndarray,
    posts_cent: np.ndarray,
    posts_rad: Union[float, int],
    tol: float = 1e-14
) -> np.ndarray:
    """
    Provide the indices of lines in contact with posts.

    Args:
        lines (np.ndarray): Array of shape (N, 4) representing pairs of points depicting lines (fibers).
                            Each row is [x1, y1, x2, y2].
        posts_cent (np.ndarray): Array of shape (M, 2) representing the 2D coordinates of post centers.
        posts_rad (Union[float, int]): Radius of all posts.
        tol (float): Tolerance for intersection checks.

    Returns:
        np.ndarray: Array of indices of lines that intersect with any post.
    """
    # Validate inputs
    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError("lines must be a 2D array with shape (N, 4).")
    if posts_cent.ndim != 2 or posts_cent.shape[1] != 2:
        raise ValueError("posts_cent must be a 2D array with shape (M, 2).")
    if not isinstance(posts_rad, (float, int)) or posts_rad <= 0:
        raise ValueError("posts_rad must be a positive number.")
    if tol < 0:
        raise ValueError("tol must be a non-negative number.")

    # Create Shapely geometries for posts and lines
    post_geometries = [Point(center).buffer(posts_rad + tol) for center in posts_cent]
    line_strings = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lines]

    # Build a spatial index for the posts
    post_index = STRtree(post_geometries)

    # Find lines that intersect with any post
    lines_on_posts_indices = []
    for i, line in enumerate(line_strings):
        # Query the spatial index for potential intersections
        intersecting_posts = post_index.query(line)

        # Check if the line intersects any post
        if any(line.intersects(post_geometries[post_ind]) for post_ind in intersecting_posts):
            lines_on_posts_indices.append(i)

    return np.array(lines_on_posts_indices)


def dict_move_items_to_another_key(
    cur_dict: dict,
    key_to_remove: int,
    key_to_add: int
) -> dict:
    """
    Move all items from the set associated with `key_to_remove` to the set associated with `key_to_add`.
    If `key_to_add` does not exist, it is created. If `key_to_remove` equals `key_to_add`, no changes are made.

    Args:
        cur_dict (dict): The dictionary containing integer keys and sets of integers as values.
        key_to_remove (int): The key whose items will be moved.
        key_to_add (int): The key to which the items will be moved.

    Returns:
        dict: The updated dictionary.
    """
    # Validate inputs
    if not isinstance(cur_dict, dict):
        raise ValueError("cur_dict must be a dictionary.")
    if not isinstance(key_to_remove, int) or not isinstance(key_to_add, int):
        raise ValueError("key_to_remove and key_to_add must be integers.")
    if key_to_remove not in cur_dict:
        raise KeyError(f"key_to_remove ({key_to_remove}) does not exist in the dictionary.")

    # If key_to_remove equals key_to_add, no changes are needed
    if key_to_remove == key_to_add:
        return cur_dict

    # Check if key_to_add exists; if not, create it
    if key_to_add not in cur_dict:
        cur_dict[key_to_add] = set()

    # Move items from key_to_remove to key_to_add
    cur_dict[key_to_add].update(cur_dict[key_to_remove])
    cur_dict[key_to_add].add(key_to_remove)  # Add key_to_remove itself to the set
    del cur_dict[key_to_remove]  # Remove key_to_remove from the dictionary

    return cur_dict


def dict_add_item_to_key(
    cur_dict: dict,
    key_new: int,
    key_add: int
) -> dict:
    """
    Add or move `key_new` to the set associated with `key_add`.
    If `key_new` is already part of another set, it is moved to `key_add`.

    Args:
        cur_dict (dict): The dictionary containing integer keys and sets of integers as values.
        key_new (int): The key to be added or moved.
        key_add (int): The key to which `key_new` will be added.

    Returns:
        dict: The updated dictionary.
    """
    # Validate inputs
    if isinstance(key_new,np.int64):
        key_new = int(key_new)
    if isinstance(key_add,np.int64):
        key_add = int(key_add)
    if not isinstance(cur_dict, dict):
        raise ValueError("cur_dict must be a dictionary.")
    if not isinstance(key_new, int) or not isinstance(key_add, int):
        raise ValueError("key_new and key_add must be integers.")

    # Check if key_add is part of a set; if so, reassign key_add to its parent key
    for key, integer_set in cur_dict.items():
        if key_add in integer_set:
            key_add = key
            break

    # If key_new equals key_add, no changes are needed
    if key_new == key_add:
        return cur_dict

    # If key_new is already a key, move its entire set to key_add
    if key_new in cur_dict:
        cur_dict = dict_move_items_to_another_key(cur_dict, key_to_remove=key_new, key_to_add=key_add)
    else:
        # Check if key_new is part of a set under another key
        key_new_found = False
        for key, integer_set in cur_dict.items():
            if key_new in integer_set:
                # Move the entire set to key_add
                cur_dict = dict_move_items_to_another_key(cur_dict, key_to_remove=key, key_to_add=key_add)
                key_new_found = True
                break

        # If key_new is not part of any set, add it to key_add
        if not key_new_found:
            if key_add in cur_dict:
                cur_dict[key_add].add(key_new)
            else:
                cur_dict[key_add] = {key_new}

    return cur_dict


def organize_dict(cur_dict: dict, lines_on_posts_ind: np.ndarray) -> dict:
    """
    Organize a dictionary by grouping keys and values based on their relationship to `lines_on_posts_ind`.

    Args:
        cur_dict (dict): A dictionary with integer keys and sets of integers as values.
        lines_on_posts_ind (np.ndarray): Array of indices representing lines on posts.

    Returns:
        dict: An organized dictionary where keys in `lines_on_posts_ind` are grouped with related keys.
    """
    # Validate inputs
    if not isinstance(cur_dict, dict):
        raise ValueError("cur_dict must be a dictionary.")
    if not isinstance(lines_on_posts_ind, np.ndarray):
        raise ValueError("lines_on_posts_ind must be a NumPy array.")

    # Initialize the organized dictionary
    organized_dict = {int(ind): set() for ind in lines_on_posts_ind}

    # Track keys not directly on posts
    keys_not_on_posts = []

    # First pass: Assign keys and values to `organized_dict`
    for key, value_set in cur_dict.items():
        if key in lines_on_posts_ind:
            organized_dict[int(key)] = value_set
        else:
            for val in value_set:
                if val in lines_on_posts_ind:
                    organized_dict[int(val)].add(key)
                    keys_not_on_posts.append(key)
                    break

    # Second pass: Handle keys not directly on posts
    for key in keys_not_on_posts:
        value_set = cur_dict[key]
        if not value_set:
            continue

        key_on_post = None
        values_to_add = set()

        for val in value_set:
            if val in lines_on_posts_ind:
                if key_on_post is None:
                    key_on_post = val
            else:
                values_to_add.add(val)

        if key_on_post is not None:
            organized_dict[int(key_on_post)].update(values_to_add)

    return organized_dict


def lines_intersect_lines_on_posts(
    lines: np.ndarray,
    lines_on_posts_ind: Union[List, np.ndarray]
) -> np.ndarray:
    """
    Provide the indices of lines that are in contact with the posts,
    or intersect with lines that are in contact with the posts.

    Args:
        lines (np.ndarray): Array of shape (N, 4) representing pairs of points depicting lines (fibers).
                            Each row is [x1, y1, x2, y2].
        lines_on_posts_ind (Union[List, np.ndarray]): Indices of lines that are directly in contact with posts.

    Returns:
        np.ndarray: Array of indices of lines that intersect with lines on posts.
    """
    # Validate inputs
    if lines.ndim != 2 or lines.shape[1] != 4:
        raise ValueError("lines must be a 2D array with shape (N, 4).")
    if not isinstance(lines_on_posts_ind, (list, np.ndarray)):
        raise ValueError("lines_on_posts_ind must be a list or NumPy array.")

    # Convert lines to Shapely LineString objects
    line_strings = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lines]

    # Initialize connectivity dictionary for lines on posts
    con_dict = {int(line_ind): set() for line_ind in lines_on_posts_ind}

    # Build a spatial index for efficient intersection queries
    spatial_index = STRtree(line_strings)

    # Process each line to find intersections
    for i, line1 in enumerate(line_strings):

        # Query the spatial index for potential intersections
        intersecting_indices = spatial_index.query(line1)

        for j in intersecting_indices:
            if i == j:
                continue  # Skip self-intersections

            line2 = line_strings[j]

            # Check if the lines intersect
            if line1.intersects(line2):
                # Add the intersecting line to the connectivity dictionary
                con_dict = dict_add_item_to_key(con_dict, key_new=j, key_add=i)

    # Organize the connectivity dictionary
    con_dict = organize_dict(con_dict, np.array(lines_on_posts_ind))

    # Collect all lines intersecting with lines on posts
    lines_con_posts_ind = set()
    for line_ind in lines_on_posts_ind:
        lines_con_posts_ind.update(con_dict[line_ind])

    # Convert the result to a NumPy array
    lines_intersect_lines_on_posts_ind = np.array(list(lines_con_posts_ind))

    return lines_intersect_lines_on_posts_ind


def remove_floating_lines(
    lines:np.ndarray,
    posts_cent:np.ndarray,
    posts_rad:Union[float,int],
    tol:float=1e-14
) -> np.ndarray:
    """
    Remove floating lines that are not in contact with posts or other lines.

    Args:
        lines (np.ndarray): Array of shape (N, 4) representing pairs of points depicting lines (fibers).
                            Each row is [x1, y1, x2, y2].
        posts_cent (np.ndarray): Array of shape (M, 2) representing the 2D coordinates of post centers.
        posts_rad (Union[float, int]): Radius of all posts.
        tol (float): Tolerance for intersection checks.

    Returns:
        np.ndarray: Array of shape (K, 4) representing the lines after removing floating lines.
    """
    # Validate inputs
    if posts_cent.ndim != 2 or posts_cent.shape[1] != 2:
        raise ValueError("posts_cent must be a 2D array with shape (M, 2).")
    if not isinstance(posts_rad, (float, int)) or posts_rad <= 0:
        raise ValueError("posts_rad must be a positive number.")
    
    # Find indices of lines in contact with posts
    lines_on_posts_ind = lines_on_posts(lines, posts_cent, posts_rad, tol=tol)

    # Find indices of lines intersecting with lines on posts
    lines_intersecting_ind = lines_intersect_lines_on_posts(lines, lines_on_posts_ind)

    # Combine and deduplicate indices
    lines_keep_ind = np.unique(np.concatenate((lines_on_posts_ind, lines_intersecting_ind))).astype(int)

    # Filter lines to keep only those with valid indices
    if lines_keep_ind.size == 0:
        raise ValueError("Ill-pose boundary conditions, tissue fiber network is not attached to pillars.")
    lines_without_floating = lines[lines_keep_ind, :]

    # Sort and remove duplicate lines
    sorted_lines = sort_points_and_lines(lines_without_floating)

    return sorted_lines


def assign_radius_to_fibers(
    lines: np.ndarray,
    mean_radii: Union[float, int],
    sd_radii: Union[float, int],
    radii_dist_type: str = 'normal',
) -> np.ndarray:
    """
    Assign radii to fibers based on the specified distribution.

    Args:
        lines (np.ndarray): Array of shape (N, 4) representing pairs of points depicting lines (fibers).
                            Each row is [x1, y1, x2, y2].
        mean_radii (Union[float, int]): Mean radius of the fibers.
        sd_radii (Union[float, int]): Standard deviation of the fiber radii (for normal distribution).
        radii_dist_type (str): Type of radius distribution ('normal', 'uniform', 'constant').

    Returns:
        np.ndarray: Array of shape (N,) representing the radii assigned to each fiber.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # Validate inputs
    if not isinstance(mean_radii, (float, int)) or mean_radii <= 0:
        raise ValueError("mean_radii must be a positive number.")
    if not isinstance(sd_radii, (float, int)) or sd_radii < 0:
        raise ValueError("sd_radii must be a non-negative number.")
    if radii_dist_type not in ['normal', 'uniform', 'constant']:
        raise ValueError("radii_dist_type must be one of 'normal', 'uniform', or 'constant'.")

    # Number of fibers
    num_fibers = lines.shape[0]

    # Handle edge case: no fibers
    if num_fibers == 0:
        return np.array([])

    # Assign radii based on the specified distribution
    if radii_dist_type == 'normal':
        # Generate radii from a normal distribution
        fib_radii = np.random.normal(mean_radii, sd_radii, num_fibers)
        # Ensure radii are positive
        fib_radii = np.clip(fib_radii, a_min=1e-10, a_max=None)
    elif radii_dist_type == 'uniform':
        # Generate radii from a uniform distribution
        min_radii = max(0, mean_radii - sd_radii)
        max_radii = mean_radii + sd_radii
        fib_radii = np.random.uniform(min_radii, max_radii, num_fibers)
    elif radii_dist_type == 'constant':
        # Assign a constant radius to all fibers
        fib_radii = np.full(num_fibers, mean_radii)

    return fib_radii


def assign_small_fib_to_original(
    og_lines: np.ndarray,
    split_lines: np.ndarray,
    old_ind: np.ndarray,
    line_tol: float = 1e-10,
) -> np.ndarray:
    """
    Assign the indices of the original lines to the split lines.

    Args:
        og_lines (np.ndarray): Array of shape (N, 4) representing the original lines.
        split_lines (np.ndarray): Array of shape (M, 4) representing the split lines.
        old_ind (np.ndarray): Array of shape (N,) containing the indices of the original lines.
        line_tol (float): Tolerance for intersection checks.

    Returns:
        np.ndarray: Array of shape (M,) containing the indices of the original lines for each split line.
    """
    # Validate inputs
    if og_lines.ndim != 2 or og_lines.shape[1] != 4:
        raise ValueError("og_lines must be a 2D array with shape (N, 4).")
    if split_lines.ndim != 2 or split_lines.shape[1] != 4:
        raise ValueError("split_lines must be a 2D array with shape (M, 4).")
    if old_ind.ndim != 1 or old_ind.shape[0] != og_lines.shape[0]:
        raise ValueError("old_ind must be a 1D array with the same length as og_lines.")
    if line_tol < 0:
        raise ValueError("line_tol must be a non-negative number.")

    # Convert lines to Shapely LineString objects
    og_strings = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in og_lines]
    split_strings = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in split_lines]

    # Build a spatial index for the original lines
    og_index = STRtree(og_strings)

    # Map split lines to original line indices
    mapping_indices = []
    for split_line in split_strings:
        # Buffer the split line to account for numerical errors
        split_line_buffered = split_line.buffer(line_tol)

        # Query the spatial index for potential candidates
        candidate_indices = og_index.query(split_line_buffered)

        # Check intersections with candidate lines
        for candidate_index in candidate_indices:
            og_line = og_strings[candidate_index]
            intersection = split_line_buffered.intersection(og_line)

            # Check if the intersection is valid
            if not intersection.is_empty:
                split_line_len = split_line.length
                intersection_len = intersection.length
                if split_line_len <= intersection_len + line_tol and intersection_len > line_tol:
                    mapping_indices.append(old_ind[candidate_index])
                    break

    return np.array(mapping_indices)


def generate_lines_for_circle(
    center: np.ndarray,
    radius: Union[float, int],
    num_lines: int = 100
) -> np.ndarray:
    """
    Generate a set of lines depicting a circle.

    Args:
        center (np.ndarray): Array of shape (2,) representing the center of the circle [x, y].
        radius (Union[float, int]): Radius of the circle.
        num_lines (int): Number of lines to approximate the circle.

    Returns:
        np.ndarray: Array of shape (num_lines, 4), where each row represents a line [x1, y1, x2, y2].

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # Validate inputs
    if center.ndim != 1 or center.shape[0] != 2:
        raise ValueError("center must be a 1D array with shape (2,).")
    if not isinstance(radius, (float, int)) or radius <= 0:
        raise ValueError("radius must be a positive number.")
    if not isinstance(num_lines, int) or num_lines <= 0:
        raise ValueError("num_lines must be a positive integer.")

    # Generate angles evenly spaced around the circle
    angles = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)

    # Calculate start points on the circle
    start_points = np.column_stack((
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ))

    # Calculate end points by shifting start points
    end_points = np.roll(start_points, shift=-1, axis=0)

    # Combine start and end points into a single array
    lines_array = np.hstack((start_points, end_points))

    # Sort and remove duplicate lines
    sorted_lines = sort_points_and_lines(lines_array)

    return sorted_lines


def add_circular_posts(
    fib_net: np.ndarray,
    posts_pos: np.ndarray,
    posts_radius: Union[float, np.ndarray],
    num_lines: int = 110,
) -> np.ndarray:
    """
    Create sets of lines depicting multiple circles (2D posts) and add them to the fiber network.

    Args:
        fib_net (np.ndarray): Array of shape (N, 4) representing the fiber network.
                              Each row is [x1, y1, x2, y2].
        posts_pos (np.ndarray): Array of shape (M, 2) representing the positions of the post centers.
        posts_radius (Union[float, np.ndarray]): Radius of the posts. Can be a scalar or an array of shape (M,).
        num_lines (int): Number of lines to approximate each circle.

    Returns:
        np.ndarray: Array of shape (N + M * num_lines, 4) representing the updated fiber network.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # Validate inputs
    if fib_net.ndim != 2 or fib_net.shape[1] != 4:
        raise ValueError("fib_net must be a 2D array with shape (N, 4).")
    if posts_pos.ndim != 2 or posts_pos.shape[1] != 2:
        raise ValueError("posts_pos must be a 2D array with shape (M, 2).")
    if not isinstance(posts_radius, (float, np.ndarray)):
        raise ValueError("posts_radius must be a float or a NumPy array.")
    if isinstance(posts_radius, np.ndarray) and posts_radius.shape[0] != posts_pos.shape[0]:
        raise ValueError("posts_radius must have the same length as posts_pos if it is an array.")
    if not isinstance(num_lines, int) or num_lines <= 0:
        raise ValueError("num_lines must be a positive integer.")

    # Ensure posts_radius is an array for consistent processing
    if isinstance(posts_radius, float) or isinstance(posts_radius, int):
        posts_radius = np.full(posts_pos.shape[0], posts_radius)

    # Generate lines for each post
    posts_lines = [
        generate_lines_for_circle(center=post_center, radius=post_radius, num_lines=num_lines)
        for post_center, post_radius in zip(posts_pos, posts_radius)
    ]

    # Concatenate all post lines into a single array
    posts_lines = np.vstack(posts_lines)

    # Combine the fiber network with the post lines
    updated_fib_net = np.vstack((fib_net, posts_lines))

    return updated_fib_net


def assign_posts_ind(
    fib_net_old: np.ndarray,
    fib_net_new: np.ndarray,
    radii_dist_type: Optional[str] = None,
    num_posts: int = 4,
    num_lines_percir: int = 110,
) -> np.ndarray:
    """
    Assign grouping indices for posts and fibers.

    Args:
        fib_net_old (np.ndarray): Array of shape (N, 4) representing the original fiber network.
        fib_net_new (np.ndarray): Array of shape (M, 4) representing the updated fiber network with posts.
        radii_dist_type (Optional[str]): Type of radius distribution (e.g., 'normal', 'uniform', etc.).
        num_posts (int): Number of posts in the network.
        num_lines_percir (int): Number of lines used to approximate each circular post.

    Returns:
        np.ndarray: Array of shape (M,) containing the indices for fibers and posts.

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # Validate inputs
    if fib_net_old.ndim != 2 or fib_net_old.shape[1] != 4:
        raise ValueError("fib_net_old must be a 2D array with shape (N, 4).")
    if fib_net_new.ndim != 2 or fib_net_new.shape[1] != 4:
        raise ValueError("fib_net_new must be a 2D array with shape (M, 4).")
    if not isinstance(num_posts, int) or num_posts <= 0:
        raise ValueError("num_posts must be a positive integer.")
    if not isinstance(num_lines_percir, int) or num_lines_percir <= 0:
        raise ValueError("num_lines_percir must be a positive integer.")

    # Calculate the number of fibers and posts
    old_num = fib_net_old.shape[0]
    new_num = fib_net_new.shape[0]
    posts_seg_num = new_num - old_num

    if posts_seg_num != num_posts * num_lines_percir:
        raise ValueError("Mismatch between the number of posts and the number of lines per post.")

    # Assign indices to the fiber network
    if radii_dist_type is None:
        fib_ind = np.ones(old_num, dtype=int) * -1  # Default index for fibers
    else:
        fib_ind = np.arange(old_num, dtype=int)  # Assign unique indices to fibers

    # Assign indices to the posts
    post_ind_offset=10000
    posts_ind = np.concatenate([
        np.ones(num_lines_percir, dtype=int) * (post_ind_offset + i) for i in range(num_posts)
    ])

    # Combine fiber and post indices
    all_ind = np.concatenate((fib_ind, posts_ind), axis=0)

    return all_ind


def generate_fib_net_for_tissue(config: FiberNetworkParams) -> FiberNetworkResult:
    """
    Create a fiber network representing the microtissue.

    Args:
        config (FiberNetworkParams): Configuration object containing all input parameters.

    Returns:
        FiberNetworkResult: Object containing the final fiber network, indices, and radii.
    """
    # Validate the configuration
    config.validate()

    # Step 1: Generate initial fibers
    all_pts = generate_initial_fibers(
        tissue_pts=config.tissue_points,
        fib_n=config.num_fibers,
        min_fib_L=config.fiber_length.get("min", 0.1),
        max_fib_L=config.fiber_length.get("max", 0.5),
        mean_fib_L=config.fiber_length.get("mean", 0.3),
        sd_fib_L=config.fiber_length.get("std_dev", 0.05),
        ang_range=config.fiber_orientation.get("range", [0, 180]),
        ang_mu=config.fiber_orientation.get("mean_angle", 0),
        ang_kappa=config.fiber_orientation.get("concentration", 1.0),
        len_dist_type=config.fiber_length.get("distribution", "uniform"),
        ang_dist_type=config.fiber_orientation.get("distribution", "von_mises"),
        num_patch_x=config.num_patches_x,
        num_patch_y=config.num_patches_y,
        dim=config.dimension,
    )

    # Step 2: Cut fibers to fit within the tissue
    cut_lines = cut_network_into_shape(all_pts, config.tissue_points)

    # Step 3: Remove fibers inside microposts
    lines_without_posts = remove_lines_in_circles(config.post_positions, config.post_radius, cut_lines)

    # Step 4: Add circular posts
    lines_with_new_posts = add_circular_posts(lines_without_posts, config.post_positions, config.post_radius)

    # Step 5: Assign indices to posts
    ind_with_posts = assign_posts_ind(lines_without_posts, lines_with_new_posts, config.fiber_radii.get("distribution"))

    # Step 6: Assign radii to fibers
    fib_radii_pre_split = assign_radius_to_fibers(
        lines_with_new_posts,
        mean_radii=config.fiber_radii.get("mean", 0.01),
        sd_radii=config.fiber_radii.get("std_dev", 0.002),
        radii_dist_type=config.fiber_radii.get("distribution", "normal"),
    )

    # Step 7: Cut fibers inside the wound
    cut_lines3 = remove_lines_in_wound(lines_with_new_posts, config.wound_shape)

    # Step 8: Remove non-intersecting fibers
    inter_fib = remove_non_intersecting_fibers(cut_lines3)

    # Step 9: Remove floating fibers
    no_floating_lines = remove_floating_lines(inter_fib, config.post_positions, config.post_radius, tol=config.post_tolerance)

    # Step 10: Split intersecting fibers into smaller fibers
    split_fib = split_large_fibers_by_intersections(no_floating_lines)

    # Step 11: Assign original indices to split fibers
    final_ind = assign_small_fib_to_original(
        og_lines=lines_with_new_posts,
        split_lines=split_fib,
        old_ind=ind_with_posts,
        line_tol=1e-10,
    )

    # Return the result as a structured object
    return FiberNetworkResult(fibers=split_fib, indices=final_ind, radii=fib_radii_pre_split)


def find_line_cells(lines: np.ndarray) -> np.ndarray:
    """
    Find unique points and map line endpoints to their indices.

    Args:
        lines (np.ndarray): Array of shape (N, 4), where each row represents a line [x1, y1, x2, y2].

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Unique points as a 2D array of shape (M, 2).
            - Line cells as a 2D array of shape (N, 2), where each row contains the indices of the line's endpoints.
    """
    if lines.shape[0]==0:
        raise ValueError("The array of lines representing the fiber network is empty.")

    # Extract all points from the lines
    all_points = lines[:, :2].tolist() + lines[:, 2:].tolist()

    # Find unique points and their indices
    all_unique_pts, _ = np.unique(all_points, axis=0, return_inverse=True)

    # Map line endpoints to their indices
    pts_tree = STRtree([Point(pt) for pt in all_unique_pts])
    line_cells = []
    for i in range(lines.shape[0]):
        p1 = Point(lines[i, 0], lines[i, 1])
        p2 = Point(lines[i, 2], lines[i, 3])

        nearest_p1_ind=pts_tree.nearest(p1)
        nearest_p2_ind=pts_tree.nearest(p2)
        line_cells.append([nearest_p1_ind, nearest_p2_ind])
    line_cells = np.array(line_cells)

    return all_unique_pts, line_cells


def match_char_len(lines: np.ndarray, char_len: Union[float, int], line_ind: np.ndarray) -> np.ndarray:
    """
    Split lines into smaller segments if their length exceeds the characteristic length.

    Args:
        lines (np.ndarray): Array of shape (N, 4), where each row represents a line [x1, y1, x2, y2].
        char_len (Union[float, int]): Maximum allowed length for a line segment.
        line_ind (np.ndarray): Array of shape (N,) containing indices corresponding to each line.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Refined lines as an array of shape (M, 4), where M >= N.
            - Updated indices as an array of shape (M,).
    """
    refined_lines = []
    new_ind = []

    for i, (p1x, p1y, p2x, p2y) in enumerate(lines):
        len_line = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)

        if len_line <= char_len:
            # Keep the line as is if it is within the characteristic length
            refined_lines.append([p1x, p1y, p2x, p2y])
            new_ind.append(line_ind[i])
        else:
            # Split the line into smaller segments
            num_seg = int(np.ceil(len_line / char_len))
            fractions = np.linspace(0, 1, num_seg + 1)

            x_vals = p1x + fractions * (p2x - p1x)
            y_vals = p1y + fractions * (p2y - p1y)

            # Create new segments
            for j in range(num_seg):
                refined_lines.append([x_vals[j], y_vals[j], x_vals[j + 1], y_vals[j + 1]])
                new_ind.append(line_ind[i])
    return np.array(refined_lines), np.array(new_ind)


def generate_fib_net_xdmf_meshio(
    fib_net: np.ndarray,
    f_name: str,
    output_xdmf: str,
    fib_ind: np.ndarray = None,
    characteristic_length: Union[float, int] = 1,
) -> None:
    """
    Generate an XDMF file for the fiber network using meshio.

    Args:
        fib_net (np.ndarray): Array of shape (N, 4) representing the fiber network.
                              Each row is [x1, y1, x2, y2].
        f_name (str): Base name for the output file.
        output_xdmf (str): Directory path for the output XDMF file.
        fib_ind (np.ndarray, optional): Array of shape (N,) containing indices for each fiber. Defaults to None.
        characteristic_length (Union[float, int]): Maximum allowed length for a line segment. Defaults to 1.

    Returns:
        None

    Raises:
        ValueError: If any input parameter is invalid.
    """
    # Validate inputs
    if fib_net.ndim != 2 or fib_net.shape[1] != 4:
        raise ValueError("fib_net must be a 2D array with shape (N, 4).")
    if not isinstance(f_name, str) or not f_name:
        raise ValueError("f_name must be a non-empty string.")
    if not isinstance(output_xdmf, str) or not output_xdmf:
        raise ValueError("output_xdmf must be a non-empty string.")
    if fib_ind is not None and (fib_ind.ndim != 1 or fib_ind.shape[0] != fib_net.shape[0]):
        raise ValueError("fib_ind must be a 1D array with the same length as fib_net.")
    if characteristic_length <= 0:
        raise ValueError("characteristic_length must be a positive number.")

    # Step 1: Refine the fiber network based on the characteristic length
    fib_net_cl, fib_ind_cl = match_char_len(
        lines=fib_net,
        char_len=characteristic_length,
        line_ind=fib_ind if fib_ind is not None else np.arange(fib_net.shape[0])
    )
    print(f"Number of fibers before accounting for characteristic length: {fib_net.shape[0]}")
    print(f"Number of fibers after accounting for characteristic length: {fib_net_cl.shape[0]}")

    # Step 2: Find unique points and line cells
    all_points, line_cells = find_line_cells(lines=fib_net_cl)

    # Step 3: Prepare line data
    line_data = fib_ind_cl

    # Step 4: Create the mesh using meshio
    fiber_mesh = meshio.Mesh(
        points=all_points,
        cells={"line": line_cells},
        cell_data={"fibers": [line_data]}
    )

    # Step 5: Write the mesh to an XDMF file
    output_file = f"{output_xdmf}{f_name}.xdmf"
    try:
        meshio.write(output_file, fiber_mesh)
        print(f"Fiber network successfully written to {output_file}")
    except Exception as e:
        raise IOError(f"Failed to write XDMF file: {e}")
