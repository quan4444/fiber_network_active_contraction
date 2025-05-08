import os
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, GeometryCollection, Point
import sys
sys.path.insert(1, 'src/fiber_network')
import mesh_generation as mg
import meshio
# import fiber_network.mesh_generation as mg


def test_hello_world(capsys):
    """
    Test the hello_world function to ensure it prints the correct message.
    """
    # Call the function
    mg.hello_world()

    # Capture the output
    captured = capsys.readouterr()

    # Assert the output is as expected
    assert captured.out.strip() == "Hello, World! This is fiber_network package."


def test_generate_sample_tissue_params():
    """
    Test the generate_sample_tissue_params function to ensure it returns valid outputs.
    """
    # Call the function
    tissue_pts, posts_pos, posts_radius, wound_shape = mg.generate_sample_tissue_params()

    # Validate tissue_pts
    assert isinstance(tissue_pts, np.ndarray), "tissue_pts should be a NumPy array."
    assert tissue_pts.shape[1] == 2, "tissue_pts should have shape (N, 2)."
    assert tissue_pts.shape[0] >= 3, "tissue_pts should have at least 3 points to form a polygon."

    # Validate posts_pos
    assert isinstance(posts_pos, np.ndarray), "posts_pos should be a NumPy array."
    assert posts_pos.shape[1] == 2, "posts_pos should have shape (M, 2)."
    assert posts_pos.shape[0] > 0, "posts_pos should have at least one post."

    # Validate posts_radius
    assert isinstance(posts_radius, (float, int)), "posts_radius should be a float or int."
    assert posts_radius > 0, "posts_radius should be a positive number."

    # Validate wound_shape
    assert isinstance(wound_shape, np.ndarray), "wound_shape should be a NumPy array."
    assert wound_shape.shape[1] == 2, "wound_shape should have shape (P, 2)."
    assert wound_shape.shape[0] >= 3, "wound_shape should have at least 3 points to form a polygon."

    # Additional checks for specific values
    assert np.allclose(tissue_pts[0], [0.0, 0.0]), "First tissue point should be [0.0, 0.0]."
    assert posts_radius == 0.1, "posts_radius should be 0.1."
    assert wound_shape.shape[0] == 100, "wound_shape should have 100 points for the ellipse."


def test_get_filename_minimal():
    """
    Test get_filename with minimal inputs.
    """
    config = mg.FiberNetworkParams()
    result = mg.get_filename(basename="fiber", fib_n=0, config=config)
    assert result == "0fiber_len_uniform_min0.1_max0.5_orien_von_mises_mu0_kappa1.0_radii_normal_mean0.01_std0.002", "Filename does not match expected minimal output."


def test_get_filename_with_uniform_length_and_orientation():
    """
    Test get_filename with uniform fiber length and orientation.
    """
    config = mg.FiberNetworkParams()
    config.fiber_length["distribution"] = "uniform"
    config.fiber_length["min"] = 0.2
    config.fiber_length["max"] = 0.8
    config.fiber_orientation["distribution"] = "uniform"
    config.fiber_orientation["range"] = [0, 90]

    result = mg.get_filename(basename="fiber", fib_n=100, config=config)
    expected = "100fiber_len_uniform_min0.2_max0.8_orien_uniform_range[0, 90]_radii_normal_mean0.01_std0.002"
    assert result == expected, "Filename does not match expected output for uniform length and orientation."


def test_get_filename_with_normal_length_and_von_mises_orientation():
    """
    Test get_filename with normal fiber length and von Mises orientation.
    """
    config = mg.FiberNetworkParams()
    config.fiber_length["distribution"] = "normal"
    config.fiber_length["mean"] = 0.5
    config.fiber_length["std_dev"] = 0.1
    config.fiber_orientation["distribution"] = "von_mises"
    config.fiber_orientation["mean_angle"] = 45
    config.fiber_orientation["concentration"] = 2.0

    result = mg.get_filename(basename="fiber", fib_n=200, config=config)
    expected = "200fiber_len_normal_mean0.5_std0.1_orien_von_mises_mu45_kappa2.0_radii_normal_mean0.01_std0.002"
    assert result == expected, "Filename does not match expected output for normal length and von Mises orientation."


def test_get_filename_with_constant_radius():
    """
    Test get_filename with constant fiber radius.
    """
    config = mg.FiberNetworkParams()
    config.fiber_radii["distribution"] = "constant"
    config.fiber_radii["mean"] = 0.02

    result = mg.get_filename(basename="fiber", fib_n=50, config=config)
    expected = "50fiber_len_uniform_min0.1_max0.5_orien_von_mises_mu0_kappa1.0_radii_constant"
    assert result == expected, "Filename does not match expected output for constant radius."


def test_get_filename_with_slit_and_seed():
    """
    Test get_filename with slit option and seed number.
    """
    config = mg.FiberNetworkParams()
    config.fiber_length["distribution"] = "uniform"
    config.fiber_length["min"] = 0.1
    config.fiber_length["max"] = 0.5

    result = mg.get_filename(
        basename="fiber",
        fib_n=300,
        config=config,
        slit_option=True,
        slit_size=0.05,
        seed_num=42,
    )
    expected = "300fiber_len_uniform_min0.1_max0.5_orien_von_mises_mu0_kappa1.0_radii_normal_mean0.01_std0.002_slit_0.05_seed42"
    assert result == expected, "Filename does not match expected output with slit and seed options."


def test_generate_random_points_in_area_basic_functionality():
    """
    Test basic functionality of generate_random_points_in_area.
    """
    shape_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    fib_n = 5
    random_points = mg.generate_random_points_in_area(shape_pts, fib_n)
    assert random_points.shape == (fib_n, 2), "Output shape is incorrect."
    assert np.all(random_points[:, 0] >= 0) and np.all(random_points[:, 0] <= 10), "X-coordinates are out of bounds."
    assert np.all(random_points[:, 1] >= 0) and np.all(random_points[:, 1] <= 10), "Y-coordinates are out of bounds."


def test_generate_random_points_in_area_invalid_shape_pts():
    """
    Test generate_random_points_in_area with invalid shape_pts (single point).
    """
    shape_pts = np.array([[5, 5]])
    fib_n = 5
    with pytest.raises(ValueError, match="shape_pts must be a 2D array with shape \\(N, 2\\), where N>=3."):
        mg.generate_random_points_in_area(shape_pts, fib_n)


def test_generate_random_points_in_area_negative_fib_n():
    """
    Test generate_random_points_in_area with a negative number of fibers.
    """
    shape_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    fib_n = -1
    with pytest.raises(ValueError, match="fib_n must be a positive integer."):
        mg.generate_random_points_in_area(shape_pts, fib_n)


def test_generate_random_points_in_area_non_rectangular_shape():
    """
    Test generate_random_points_in_area with a non-rectangular shape.
    """
    shape_pts = np.array([[0, 0], [5, 10], [10, 0]])
    fib_n = 10
    random_points = mg.generate_random_points_in_area(shape_pts, fib_n)
    assert random_points.shape == (fib_n, 2), "Output shape is incorrect for non-rectangular shape."
    assert np.all(random_points[:, 0] >= 0) and np.all(random_points[:, 0] <= 10), "X-coordinates are out of bounds for non-rectangular shape."
    assert np.all(random_points[:, 1] >= 0) and np.all(random_points[:, 1] <= 10), "Y-coordinates are out of bounds for non-rectangular shape."


def test_generate_points_by_patches_valid_input():
    """
    Test the function with valid inputs.
    """
    tissue_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    fib_n = 100
    num_patch_x = 2
    num_patch_y = 2

    points = mg.generate_points_by_patches(tissue_pts, fib_n, num_patch_x, num_patch_y)

    # Check the shape of the output
    assert points.shape == (fib_n, 2), "Output shape is incorrect."

    # Check that all points are within the bounding box
    assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] <= 10), "X-coordinates are out of bounds."
    assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] <= 10), "Y-coordinates are out of bounds."


def test_generate_points_by_patches_invalid_tissue_pts():
    """
    Test the function with invalid tissue_pts input.
    """
    tissue_pts = np.array([[0, 0]])  # Invalid: less than 3 points
    fib_n = 100
    num_patch_x = 2
    num_patch_y = 2

    with pytest.raises(ValueError, match="tissue_pts must be a 2D array with shape \\(N, 2\\), where N>=3."):
        mg.generate_points_by_patches(tissue_pts, fib_n, num_patch_x, num_patch_y)


def test_generate_points_by_patches_invalid_fib_n():
    """
    Test the function with invalid fib_n input.
    """
    tissue_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    fib_n = -10  # Invalid: negative number of fibers
    num_patch_x = 2
    num_patch_y = 2

    with pytest.raises(ValueError, match="fib_n must be a positive integer."):
        mg.generate_points_by_patches(tissue_pts, fib_n, num_patch_x, num_patch_y)


def test_generate_points_by_patches_invalid_num_patches():
    """
    Test the function with invalid num_patch_x and num_patch_y inputs.
    """
    tissue_pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    fib_n = 100
    num_patch_x = 0  # Invalid: non-positive number of patches
    num_patch_y = -1  # Invalid: non-positive number of patches

    with pytest.raises(ValueError, match="num_patch_x and num_patch_y must be positive integers."):
        mg.generate_points_by_patches(tissue_pts, fib_n, num_patch_x, num_patch_y)


def test_generate_points_by_patches_non_rectangular_tissue():
    """
    Test the function with a non-rectangular tissue shape.
    """
    tissue_pts = np.array([[0, 0], [5, 10], [10, 0]])  # Triangle
    fib_n = 50
    num_patch_x = 2
    num_patch_y = 2

    points = mg.generate_points_by_patches(tissue_pts, fib_n, num_patch_x, num_patch_y)

    # Check the shape of the output
    assert points.shape == (fib_n, 2), "Output shape is incorrect for non-rectangular tissue."

    # Check that all points are within the bounding box
    min_x, min_y = np.amin(tissue_pts, axis=0)
    max_x, max_y = np.amax(tissue_pts, axis=0)
    assert np.all(points[:, 0] >= min_x) and np.all(points[:, 0] <= max_x), "X-coordinates are out of bounds."
    assert np.all(points[:, 1] >= min_y) and np.all(points[:, 1] <= max_y), "Y-coordinates are out of bounds."


def test_generate_fiber_lengths_constant():
    fib_n = 5
    result = mg.generate_fiber_lengths(fib_n, 'constant', min_fib_L=2, max_fib_L=2, mean_fib_L=0, sd_fib_L=0)
    assert result.shape == (fib_n, 1)
    assert np.all(result == 2), "All lengths should be constant and equal to 2."


def test_generate_fiber_lengths_uniform():
    fib_n = 10
    result = mg.generate_fiber_lengths(fib_n, 'uniform', min_fib_L=1, max_fib_L=5, mean_fib_L=0, sd_fib_L=0)
    assert result.shape == (fib_n, 1)
    assert np.all(result >= 1) and np.all(result <= 5), "Lengths should be within the uniform range."


def test_generate_fiber_lengths_normal():
    fib_n = 1000
    result = mg.generate_fiber_lengths(fib_n, 'normal', min_fib_L=0, max_fib_L=0, mean_fib_L=10, sd_fib_L=2)
    assert result.shape == (fib_n, 1)
    assert np.isclose(np.mean(result), 10, atol=0.5), "Mean length should be close to 10."
    assert np.isclose(np.std(result), 2, atol=0.5), "Standard deviation should be close to 2."


def test_generate_fiber_lengths_invalid():
    with pytest.raises(ValueError, match="Invalid length distribution type: invalid"):
        mg.generate_fiber_lengths(5, 'invalid', min_fib_L=0, max_fib_L=0, mean_fib_L=0, sd_fib_L=0)


def test_normalize_vectors():
    vectors = np.array([[3, 4], [1, 0], [0, 0]])
    result = mg.normalize_vectors(vectors)
    assert result.shape == vectors.shape
    assert np.isclose(np.linalg.norm(result[0]), 1), "First vector should be normalized."
    assert np.isclose(np.linalg.norm(result[1]), 1), "Second vector should be normalized."
    assert np.all(result[2] == 0), "Zero vector should remain zero."


def test_randomize_directions():
    vectors = np.array([[1, 1], [2, 2], [3, 3]])
    result = mg.randomize_directions(vectors, dim=2)
    assert result.shape == vectors.shape
    assert np.all(np.abs(result) == np.abs(vectors)), "Randomized vectors should have the same magnitudes."


def test_generate_random_angles_random_orientation():
    fib_n = 5
    result = mg.generate_random_angles(fib_n, 'random_orientation', ang_range=[], ang_mu=0, ang_kappa=0, dim=2)
    assert result.shape == (fib_n, 2)
    assert np.allclose(np.linalg.norm(result, axis=1), 1), "All vectors should be unit vectors."


def test_generate_random_angles_uniform():
    fib_n = 5
    result = mg.generate_random_angles(fib_n, 'uniform', ang_range=[0, 90], ang_mu=0, ang_kappa=0, dim=2)
    assert result.shape == (fib_n, 2)
    assert np.allclose(np.linalg.norm(result, axis=1), 1), "All vectors should be unit vectors."


def test_generate_random_angles_von_mises():
    fib_n = 5
    result = mg.generate_random_angles(fib_n, 'von_mises', ang_range=[], ang_mu=0, ang_kappa=1, dim=2)
    assert result.shape == (fib_n, 2)
    assert np.allclose(np.linalg.norm(result, axis=1), 1), "All vectors should be unit vectors."


def test_generate_random_angles_invalid():
    with pytest.raises(ValueError, match="Invalid angle distribution type: invalid"):
        mg.generate_random_angles(5, 'invalid', ang_range=[], ang_mu=0, ang_kappa=0, dim=2)


def test_generate_fib_second_pts_valid():
    first_pts = np.array([[0, 0], [1, 1]])
    result = mg.generate_fib_second_pts(
        first_pts=first_pts,
        min_fib_L=1,
        max_fib_L=2,
        mean_fib_L=1.5,
        sd_fib_L=0.1,
        ang_range=[0, 90],
        ang_mu=0,
        ang_kappa=1,
        len_dist_type='uniform',
        ang_dist_type='von_mises',
        dim=2
    )
    assert result.shape == (2, 2)
    assert not np.allclose(result, first_pts), "Second points should differ from first points."


def test_generate_fib_second_pts_invalid_dim():
    first_pts = np.array([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValueError, match="first_pts must be a 2D array with shape \\(N, 2\\)."):
        mg.generate_fib_second_pts(
            first_pts=first_pts,
            min_fib_L=1,
            max_fib_L=2,
            mean_fib_L=1.5,
            sd_fib_L=0.1,
            ang_range=[0, 90],
            ang_mu=0,
            ang_kappa=1,
            len_dist_type='uniform',
            ang_dist_type='von_mises',
            dim=2
        )


def test_generate_initial_fibers_valid_input():
    """
    Test that the function generates fibers with valid input.
    """
    tissue_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Square tissue
    fib_n = 10
    min_fib_L = 0.1
    max_fib_L = 0.5
    mean_fib_L = 0.3
    sd_fib_L = 0.1
    ang_range = [0, 180]
    ang_mu = 0
    ang_kappa = 1
    len_dist_type = 'uniform'
    ang_dist_type = 'von_mises'
    num_patch_x = 2
    num_patch_y = 2
    dim = 2

    fibers = mg.generate_initial_fibers(
        tissue_pts=tissue_pts,
        fib_n=fib_n,
        min_fib_L=min_fib_L,
        max_fib_L=max_fib_L,
        mean_fib_L=mean_fib_L,
        sd_fib_L=sd_fib_L,
        ang_range=ang_range,
        ang_mu=ang_mu,
        ang_kappa=ang_kappa,
        len_dist_type=len_dist_type,
        ang_dist_type=ang_dist_type,
        num_patch_x=num_patch_x,
        num_patch_y=num_patch_y,
        dim=dim,
    )

    # Check output shape
    assert fibers.shape == (fib_n, 2 * dim), "Output shape is incorrect."

    # Check that fiber lengths are within the specified range
    lengths = np.linalg.norm(fibers[:, dim:] - fibers[:, :dim], axis=1)
    assert np.all(lengths >= min_fib_L), "Some fibers are shorter than the minimum length."
    assert np.all(lengths <= max_fib_L), "Some fibers are longer than the maximum length."


def test_generate_initial_fibers_invalid_tissue_pts():
    """
    Test that the function raises an error for invalid tissue_pts.
    """
    with pytest.raises(ValueError, match="tissue_pts must be a 2D array with shape"):
        mg.generate_initial_fibers(
            tissue_pts=np.array([0, 1]),  # Invalid shape
            fib_n=10,
            min_fib_L=0.1,
            max_fib_L=0.5,
            mean_fib_L=0.3,
            sd_fib_L=0.1,
            ang_range=[0, 180],
            ang_mu=0,
            ang_kappa=1,
            len_dist_type='uniform',
            ang_dist_type='von_mises',
            num_patch_x=2,
            num_patch_y=2,
            dim=2,
        )


def test_generate_initial_fibers_invalid_fib_n():
    """
    Test that the function raises an error for invalid fib_n.
    """
    tissue_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Square tissue
    with pytest.raises(ValueError, match="fib_n must be a positive integer"):
        mg.generate_initial_fibers(
            tissue_pts=tissue_pts,
            fib_n=-5,  # Invalid number of fibers
            min_fib_L=0.1,
            max_fib_L=0.5,
            mean_fib_L=0.3,
            sd_fib_L=0.1,
            ang_range=[0, 180],
            ang_mu=0,
            ang_kappa=1,
            len_dist_type='uniform',
            ang_dist_type='von_mises',
            num_patch_x=2,
            num_patch_y=2,
            dim=2,
        )


def test_generate_initial_fibers_invalid_len_dist_type():
    """
    Test that the function raises an error for invalid len_dist_type.
    """
    tissue_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Square tissue
    with pytest.raises(ValueError, match="Invalid len_dist_type"):
        mg.generate_initial_fibers(
            tissue_pts=tissue_pts,
            fib_n=10,
            min_fib_L=0.1,
            max_fib_L=0.5,
            mean_fib_L=0.3,
            sd_fib_L=0.1,
            ang_range=[0, 180],
            ang_mu=0,
            ang_kappa=1,
            len_dist_type='invalid_type',  # Invalid length distribution type
            ang_dist_type='von_mises',
            num_patch_x=2,
            num_patch_y=2,
            dim=2,
        )


def test_generate_initial_fibers_edge_case():
    """
    Test edge cases with minimal inputs.
    """
    tissue_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Square tissue
    fib_n = 1
    min_fib_L = 0.1
    max_fib_L = 0.1  # Fixed length
    mean_fib_L = 0.1
    sd_fib_L = 0.0
    ang_range = [0, 180]
    ang_mu = 0
    ang_kappa = 1
    len_dist_type = 'constant'
    ang_dist_type = 'uniform'
    num_patch_x = 1
    num_patch_y = 1
    dim = 2

    fibers = mg.generate_initial_fibers(
        tissue_pts=tissue_pts,
        fib_n=fib_n,
        min_fib_L=min_fib_L,
        max_fib_L=max_fib_L,
        mean_fib_L=mean_fib_L,
        sd_fib_L=sd_fib_L,
        ang_range=ang_range,
        ang_mu=ang_mu,
        ang_kappa=ang_kappa,
        len_dist_type=len_dist_type,
        ang_dist_type=ang_dist_type,
        num_patch_x=num_patch_x,
        num_patch_y=num_patch_y,
        dim=dim,
    )

    # Check output shape
    assert fibers.shape == (fib_n, 2 * dim), "Output shape is incorrect for edge case."

    # Check that fiber length is exactly the fixed length
    length = np.linalg.norm(fibers[:, dim:] - fibers[:, :dim], axis=1)[0]
    assert length == pytest.approx(min_fib_L), "Fiber length does not match the fixed length."


def test_cut_network_into_shape_valid_input():
    """
    Test that the function correctly cuts lines with valid input.
    """
    # Define a square shape
    shape_vertices = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ])

    # Define lines
    lines = np.array([
        [0.1, 0.1, 0.9, 0.9],  # Fully inside
        [0.5, 0.5, 1.5, 1.5],  # Partially outside
        [1.1, 1.1, 2.0, 2.0],  # Fully outside
    ])

    # Call the function
    cut_lines = mg.cut_network_into_shape(lines, shape_vertices)

    # Expected output
    expected_cut_lines = np.array([
        [0.1, 0.1, 0.9, 0.9],  # Fully inside
        [0.5, 0.5, 1.0, 1.0],  # Cut at the boundary
    ])

    # Check the result
    assert cut_lines.shape == expected_cut_lines.shape, "Output shape is incorrect."
    assert np.allclose(cut_lines, expected_cut_lines), "Output values are incorrect."


def test_cut_network_into_shape_no_intersection():
    """
    Test that the function returns an empty array when no lines intersect the shape.
    """
    # Define a square shape
    shape_vertices = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ])

    # Define lines completely outside the shape
    lines = np.array([
        [1.1, 1.1, 2.0, 2.0],
        [2.0, 2.0, 3.0, 3.0],
    ])

    # Call the function
    cut_lines = mg.cut_network_into_shape(lines, shape_vertices)

    # Check the result
    assert cut_lines.shape == (0, 4), "Output should be an empty array when no lines intersect."


def test_cut_network_into_shape_all_inside():
    """
    Test that the function returns the same lines when all lines are fully inside the shape.
    """
    # Define a square shape
    shape_vertices = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ])

    # Define lines fully inside the shape
    lines = np.array([
        [0.1, 0.1, 0.9, 0.9],
        [0.2, 0.2, 0.8, 0.8],
    ])

    # Call the function
    cut_lines = mg.cut_network_into_shape(lines, shape_vertices)

    # Check the result
    assert cut_lines.shape == lines.shape, "Output shape is incorrect."
    assert np.allclose(cut_lines, lines), "Output values should match the input when all lines are inside."


def test_cut_network_into_shape_invalid_lines_shape():
    """
    Test that the function raises an error for invalid lines input shape.
    """
    # Define a square shape
    shape_vertices = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ])

    # Define invalid lines input
    invalid_lines = np.array([
        [0.1, 0.1, 0.9],  # Incorrect shape
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines must be a 2D array with shape"):
        mg.cut_network_into_shape(invalid_lines, shape_vertices)


def test_cut_network_into_shape_invalid_shape_vertices():
    """
    Test that the function raises an error for invalid shape_vertices input.
    """
    # Define invalid shape_vertices input
    invalid_shape_vertices = np.array([
        [0, 0],
        [1, 0],
    ])  # Not enough points to form a polygon

    # Define valid lines
    lines = np.array([
        [0.1, 0.1, 0.9, 0.9],
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="shape_vertices must be a 2D array with shape"):
        mg.cut_network_into_shape(lines, invalid_shape_vertices)


def test_cut_network_into_shape_invalid_polygon():
    """
    Test that the function raises an error when shape_vertices do not form a valid polygon.
    """
    # Define invalid shape_vertices (self-intersecting polygon)
    invalid_shape_vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ])

    # Define valid lines
    lines = np.array([
        [0.1, 0.1, 0.9, 0.9],
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="shape_vertices do not form a valid polygon"):
        mg.cut_network_into_shape(lines, invalid_shape_vertices)


def test_cut_network_into_shape_edge_case_degenerate_polygon():
    """
    Test that the function handles a degenerate polygon (e.g., a line or a single point).
    """
    # Define a degenerate polygon (a single point)
    degenerate_shape_vertices = np.array([
        [0, 0],
    ])

    # Define valid lines
    lines = np.array([
        [0.1, 0.1, 0.9, 0.9],
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="shape_vertices must be a 2D array with shape"):
        mg.cut_network_into_shape(lines, degenerate_shape_vertices)


def test_remove_lines_in_circles_valid_input():
    """
    Test that the function correctly removes lines inside circles with valid input.
    """
    # Define circle centers and radius
    circle_centers = np.array([[1.0, 1.0]])
    circle_radius = 0.5

    # Define lines
    lines = np.array([
        [1.0, 1.0, 1.2, 1.2],  # Completely inside the circle
        [0.0, 1.0, 1.0, 1.0],  # Partially inside the circle
        [0,0,1,0],  # Completely outside all circles
    ])

    # Call the function
    resulting_lines = mg.remove_lines_in_circles(circle_centers, circle_radius, lines)

    # Expected output
    expected_lines = np.array([
        [0,0,1,0],
        [0.0,1.0,0.5,1.0],
    ])

    # Check the result
    assert resulting_lines.shape == expected_lines.shape, "Output shape is incorrect."
    assert np.allclose(resulting_lines, expected_lines), "Output values are incorrect."


def test_remove_lines_in_circles_no_intersection():
    """
    Test that the function returns the same lines when no lines intersect any circles.
    """
    # Define circle centers and radius
    circle_centers = np.array([[0.5, 0.5]])
    circle_radius = 0.1

    # Define lines completely outside the circle
    lines = np.array([
        [0.6, 0.6, 0.7, 0.7],
        [1.0, 1.0, 2.0, 2.0],
    ])

    # Call the function
    resulting_lines = mg.remove_lines_in_circles(circle_centers, circle_radius, lines)

    # Check the result
    assert resulting_lines.shape == lines.shape, "Output shape is incorrect when no lines intersect."
    assert np.allclose(resulting_lines, lines), "Output values should match the input when no lines intersect."


def test_remove_lines_in_circles_all_inside():
    """
    Test that the function returns an empty array when all lines are fully inside the circles.
    """
    # Define circle centers and radius
    circle_centers = np.array([[0.5, 0.5]])
    circle_radius = 1.0

    # Define lines fully inside the circle
    lines = np.array([
        [0.1, 0.1, 0.2, 0.2],
        [0.3, 0.3, 0.4, 0.4],
    ])

    # Call the function
    resulting_lines = mg.remove_lines_in_circles(circle_centers, circle_radius, lines)

    # Check the result
    assert resulting_lines.shape == (0,), "Output should be an empty array when all lines are inside."


def test_remove_lines_in_circles_invalid_circle_centers():
    """
    Test that the function raises an error for invalid circle_centers input.
    """
    # Define invalid circle_centers
    invalid_circle_centers = np.array([0.5, 0.5])  # Incorrect shape

    # Define valid lines
    lines = np.array([
        [0.1, 0.1, 0.9, 0.9],
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="circle_centers must be a 2D array with shape"):
        mg.remove_lines_in_circles(invalid_circle_centers, 0.5, lines)


def test_remove_lines_in_circles_invalid_circle_radius():
    """
    Test that the function raises an error for invalid circle_radius input.
    """
    # Define valid circle_centers
    circle_centers = np.array([[0.5, 0.5]])

    # Define valid lines
    lines = np.array([
        [0.1, 0.1, 0.9, 0.9],
    ])

    # Call the function with invalid circle_radius
    with pytest.raises(ValueError, match="circle_radius must be a positive number"):
        mg.remove_lines_in_circles(circle_centers, -0.5, lines)  # Negative radius


def test_remove_lines_in_circles_invalid_lines():
    """
    Test that the function raises an error for invalid lines input.
    """
    # Define valid circle_centers and radius
    circle_centers = np.array([[0.5, 0.5]])
    circle_radius = 0.5

    # Define invalid lines input
    invalid_lines = np.array([
        [0.1, 0.1, 0.9],  # Incorrect shape
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines must be a 2D array with shape"):
        mg.remove_lines_in_circles(circle_centers, circle_radius, invalid_lines)


def test_convert_linestring_multilinestring_to_nparray_linestring():
    """
    Test that the function correctly converts a list of LineString objects to a NumPy array.
    """
    # Define LineString objects
    line1 = LineString([(0, 0), (1, 1)])
    line2 = LineString([(1, 1), (2, 2)])

    # Call the function
    result = mg.convert_linestring_multilinestring_to_nparray([line1, line2])

    # Expected output
    expected = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."


def test_convert_linestring_multilinestring_to_nparray_multilinestring():
    """
    Test that the function correctly converts a MultiLineString object to a NumPy array.
    """
    # Define a MultiLineString object
    multi_line = MultiLineString([
        [(0, 0), (1, 1)],
        [(1, 1), (2, 2)],
    ])

    # Call the function
    result = mg.convert_linestring_multilinestring_to_nparray([multi_line])

    # Expected output
    expected = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."


def test_convert_linestring_multilinestring_to_nparray_geometry_collection():
    """
    Test that the function correctly converts a GeometryCollection object to a NumPy array.
    """
    # Define a GeometryCollection object
    geometry_collection = GeometryCollection([
        LineString([(0, 0), (1, 1)]),
        LineString([(1, 1), (2, 2)]),
    ])

    # Call the function
    result = mg.convert_linestring_multilinestring_to_nparray([geometry_collection])

    # Expected output
    expected = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."


def test_remove_non_intersecting_fibers_valid_input():
    """
    Test that the function correctly removes non-intersecting lines with valid input.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Intersects with the second line
        [1, 1, 2, 2],  # Intersects with the first line
        [3, 3, 4, 4],  # Does not intersect with any line
    ])

    # Call the function
    result = mg.remove_non_intersecting_fibers(lines)

    # Expected output
    expected = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."


def test_remove_non_intersecting_fibers_no_intersections():
    """
    Test that the function returns an empty array when no lines intersect.
    """
    # Define lines with no intersections
    lines = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [4, 4, 5, 5],
    ])

    # Call the function
    result = mg.remove_non_intersecting_fibers(lines)

    # Check the result
    assert result.shape == (0,), "Output should be an empty array when no lines intersect."


def test_remove_non_intersecting_fibers_all_intersect():
    """
    Test that the function returns the same lines when all lines intersect with at least one other line.
    """
    # Define lines where all intersect
    lines = np.array([
        [0, 0, 1, 1],  # Intersects with the second line
        [1, 1, 2, 2],  # Intersects with the first and third lines
        [2, 2, 3, 3],  # Intersects with the second line
    ])

    # Call the function
    result = mg.remove_non_intersecting_fibers(lines)

    # Check the result
    assert result.shape == lines.shape, "Output shape is incorrect when all lines intersect."
    assert np.allclose(result, lines), "Output values should match the input when all lines intersect."


def test_remove_non_intersecting_fibers_single_line():
    """
    Test that the function returns an empty array when there is only one line.
    """
    # Define a single line
    lines = np.array([
        [0, 0, 1, 1],
    ])

    # Call the function
    result = mg.remove_non_intersecting_fibers(lines)

    # Check the result
    assert result.shape == (0,), "Output should be an empty array for a single line."


def test_remove_non_intersecting_fibers_empty_input():
    """
    Test that the function returns an empty array when the input is empty.
    """
    # Define empty input
    lines = np.empty((0, 4))

    # Call the function
    result = mg.remove_non_intersecting_fibers(lines)

    # Check the result
    assert result.shape == (0,), "Output should be an empty array for empty input."


def test_remove_non_intersecting_fibers_invalid_input_shape():
    """
    Test that the function raises an error for invalid input shape.
    """
    # Define invalid input
    invalid_lines = np.array([
        [0, 0, 1],  # Incorrect shape
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines must be a 2D array with shape"):
        mg.remove_non_intersecting_fibers(invalid_lines)


def test_remove_non_intersecting_fibers_invalid_input_type():
    """
    Test that the function raises an error for invalid input type.
    """
    # Define invalid input
    invalid_lines = "invalid_input"

    # Call the function and check for ValueError
    with pytest.raises(AttributeError):
        mg.remove_non_intersecting_fibers(invalid_lines)


def test_remove_lines_in_wound_valid_input():
    """
    Test that the function correctly removes lines inside the wound with valid input.
    """
    # Define lines
    lines = np.array([
        [0.1, 0.1, 1.2, 0.1],  # Partially inside the wound
        [1.0, 1.0, 2.0, 2.0],  # Completely outside the wound
        [0.2, 0.2, 0.3, 0.3],  # Completely inside the wound
    ])

    # Define wound shape
    shape_vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    # Call the function
    result = mg.remove_lines_in_wound(lines, shape_vertices)

    # Expected output
    expected = np.array([
        [1.0, 0.1, 1.2, 0.1],  # Portion outside the wound
        [1.0, 1.0, 2.0, 2.0],  # Completely outside the wound
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."


def test_remove_lines_in_wound_no_intersections():
    """
    Test that the function returns the same lines when no lines intersect the wound.
    """
    # Define lines
    lines = np.array([
        [1.1, 1.1, 2.0, 2.0],  # Completely outside the wound
        [2.1, 2.1, 3.0, 3.0],  # Completely outside the wound
    ])

    # Define wound shape
    shape_vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    # Call the function
    result = mg.remove_lines_in_wound(lines, shape_vertices)

    # Check the result
    assert result.shape == lines.shape, "Output shape is incorrect when no lines intersect."
    assert np.allclose(result, lines), "Output values should match the input when no lines intersect."


def test_remove_lines_in_wound_all_inside():
    """
    Test that the function returns an empty array when all lines are fully inside the wound.
    """
    # Define lines
    lines = np.array([
        [0.1, 0.1, 0.2, 0.2],  # Completely inside the wound
        [0.3, 0.3, 0.4, 0.4],  # Completely inside the wound
    ])

    # Define wound shape
    shape_vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    # Call the function
    result = mg.remove_lines_in_wound(lines, shape_vertices)

    # Check the result
    assert result.shape == (0,), "Output should be an empty array when all lines are inside."


def test_remove_lines_in_wound_invalid_lines_shape():
    """
    Test that the function raises an error for invalid lines input shape.
    """
    # Define invalid lines
    invalid_lines = np.array([
        [0.1, 0.1, 0.2],  # Incorrect shape
    ])

    # Define wound shape
    shape_vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines must be a 2D array with shape"):
        mg.remove_lines_in_wound(invalid_lines, shape_vertices)


def test_remove_lines_in_wound_invalid_shape_vertices():
    """
    Test that the function raises an error for invalid shape_vertices input.
    """
    # Define lines
    lines = np.array([
        [0.1, 0.1, 0.2, 0.2],
    ])

    # Define invalid shape_vertices
    invalid_shape_vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
    ])  # Not enough points to form a polygon

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="shape_vertices must be a 2D array with shape"):
        mg.remove_lines_in_wound(lines, invalid_shape_vertices)


def test_remove_lines_in_wound_invalid_polygon():
    """
    Test that the function raises an error when shape_vertices do not form a valid polygon.
    """
    # Define lines
    lines = np.array([
        [0.1, 0.1, 0.2, 0.2],
    ])

    # Define invalid shape_vertices (self-intersecting polygon)
    invalid_shape_vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="shape_vertices do not form a valid polygon"):
        mg.remove_lines_in_wound(lines, invalid_shape_vertices)


def test_remove_lines_in_wound_edge_case_degenerate_polygon():
    """
    Test that the function handles a degenerate polygon (e.g., a line or a single point).
    """
    # Define lines
    lines = np.array([
        [0.1, 0.1, 0.2, 0.2],
    ])

    # Define a degenerate polygon (a single point)
    degenerate_shape_vertices = np.array([
        [0.0, 0.0],
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="shape_vertices must be a 2D array with shape"):
        mg.remove_lines_in_wound(lines, degenerate_shape_vertices)


def test_sort_points_and_lines_unsorted_lines():
    """
    Test that the function correctly sorts the endpoints of unsorted lines.
    """
    # Define unsorted lines
    lines = np.array([
        [1, 1, 0, 0],  # Reversed line
        [2, 2, 3, 3],  # Already sorted
        [4, 5, 3, 2],  # Unsorted line
    ])

    # Call the function
    result = mg.sort_points_and_lines(lines)

    # Expected output
    expected = np.array([
        [0, 0, 1, 1],  # Sorted
        [2, 2, 3, 3],  # Already sorted
        [3, 2, 4, 5],  # Sorted
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."


def test_sort_points_and_lines_with_duplicates():
    """
    Test that the function removes duplicate lines, including reversed duplicates.
    """
    # Define lines with duplicates
    lines = np.array([
        [0, 0, 1, 1],
        [1, 1, 0, 0],  # Reversed duplicate
        [2, 2, 3, 3],
        [3, 3, 2, 2],  # Reversed duplicate
    ])

    # Call the function
    result = mg.sort_points_and_lines(lines)

    # Expected output
    expected = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."


def test_sort_points_and_lines_single_line():
    """
    Test that the function correctly handles a single line.
    """
    # Define a single line
    lines = np.array([
        [1, 1, 0, 0],  # Reversed line
    ])

    # Call the function
    result = mg.sort_points_and_lines(lines)

    # Expected output
    expected = np.array([
        [0, 0, 1, 1],  # Sorted
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect for a single line."
    assert np.allclose(result, expected), "Output values are incorrect for a single line."


def test_sort_points_and_lines_empty_input():
    """
    Test that the function returns an empty array when the input is empty.
    """
    # Define empty input
    lines = np.empty((0, 4))

    # Call the function
    result = mg.sort_points_and_lines(lines)

    # Check the result
    assert result.shape == (0, ), "Output should be an empty array for empty input."


def test_sort_points_and_lines_invalid_input_shape():
    """
    Test that the function raises an error for invalid input shape.
    """
    # Define invalid input
    invalid_lines = np.array([
        [0, 0, 1],  # Incorrect shape
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines must be a 2D array with shape"):
        mg.sort_points_and_lines(invalid_lines)


def test_sort_points_and_lines_invalid_input_type():
    """
    Test that the function raises an error for invalid input type.
    """
    # Define invalid input
    invalid_lines = "invalid_input"

    # Call the function and check for ValueError
    with pytest.raises(AttributeError):
        mg.sort_points_and_lines(invalid_lines)


def test_sort_points_and_lines_edge_case():
    """
    Test an edge case that failed previously. Already sorted lines with no duplicates.
    """
    lines = np.array([
       [0., 0., 1., 1.],
       [0., 2., 1., 1.],
       [1., 1., 2., 0.],
       [1., 1., 2., 2.]]
       )

    # Call the function
    result = mg.sort_points_and_lines(lines)

    # Expected output
    expected = np.array([
       [0., 0., 1., 1.],
       [0., 2., 1., 1.],
       [1., 1., 2., 0.],
       [1., 1., 2., 2.]]
       )

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."
 

def test_split_large_fibers_by_intersections_with_intersections():
    """
    Test that the function correctly splits fibers at intersections.
    """
    # Define lines with intersections
    lines = np.array([
        [0, 0, 2, 2],  # Intersects with the second line
        [0, 2, 2, 0],  # Intersects with the first line
    ])

    # Call the function
    result = mg.split_large_fibers_by_intersections(lines, tolerance=0.1)

    # Expected output: lines split at the intersection point (1, 1)
    expected = np.array([
        [0, 0, 1, 1],
        [0, 2, 1, 1],
        [1, 1, 2, 0],
        [1, 1, 2, 2],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."


def test_split_large_fibers_by_intersections_no_intersections():
    """
    Test that the function returns the same lines when there are no intersections.
    """
    # Define lines with no intersections
    lines = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3],
    ])

    # Call the function
    result = mg.split_large_fibers_by_intersections(lines, tolerance=0.1)

    # Check the result
    assert result.shape == lines.shape, "Output shape is incorrect when there are no intersections."
    assert np.allclose(result, lines), "Output values should match the input when there are no intersections."


def test_split_large_fibers_by_intersections_single_line():
    """
    Test that the function correctly handles a single line.
    """
    # Define a single line
    lines = np.array([
        [0, 0, 1, 1],
    ])

    # Call the function
    result = mg.split_large_fibers_by_intersections(lines, tolerance=0.1)

    # Expected output: the same single line
    expected = np.array([
        [0, 0, 1, 1],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect for a single line."
    assert np.allclose(result, expected), "Output values are incorrect for a single line."


def test_split_large_fibers_by_intersections_empty_input():
    """
    Test that the function returns an empty array when the input is empty.
    """
    # Define empty input
    lines = np.empty((0, 4))

    # Call the function
    result = mg.split_large_fibers_by_intersections(lines, tolerance=0.1)

    # Check the result
    assert result.shape == (0,), "Output should be an empty array for empty input."


def test_split_large_fibers_by_intersections_invalid_input_shape():
    """
    Test that the function raises an error for invalid input shape.
    """
    # Define invalid input
    invalid_lines = np.array([
        [0, 0, 1],  # Incorrect shape
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines must be a 2D array with shape"):
        mg.split_large_fibers_by_intersections(invalid_lines, tolerance=0.1)


def test_split_large_fibers_by_intersections_invalid_tolerance():
    """
    Test that the function raises an error for invalid tolerance.
    """
    # Define valid lines
    lines = np.array([
        [0, 0, 1, 1],
    ])

    # Call the function with invalid tolerance
    with pytest.raises(ValueError, match="tolerance must be a positive number."):
        mg.split_large_fibers_by_intersections(lines, tolerance=0)


def test_create_slit_wound_horizontal():
    """
    Test that the function correctly creates a horizontal slit wound.
    """
    # Define input
    slit = np.array([
        [0.5, 0.5],  # Start point
        [1.0, 0.5],  # End point
    ])
    slit_dir = 'x'
    slit_dist_from_mid = 0.1

    # Call the function
    result = mg.create_slit_wound(slit, slit_dir, slit_dist_from_mid)

    # Expected output
    expected = np.array([
        [0.4, 0.5],
        [0.6, 0.5],
        [1.1, 0.5],
        [0.9, 0.5],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect for horizontal slit."
    assert np.allclose(result, expected), "Output values are incorrect for horizontal slit."


def test_create_slit_wound_vertical():
    """
    Test that the function correctly creates a vertical slit wound.
    """
    # Define input
    slit = np.array([
        [0.5, 0.5],  # Start point
        [0.5, 1.0],  # End point
    ])
    slit_dir = 'y'
    slit_dist_from_mid = 0.1

    # Call the function
    result = mg.create_slit_wound(slit, slit_dir, slit_dist_from_mid)

    # Expected output
    expected = np.array([
        [0.5, 0.4],
        [0.5, 0.6],
        [0.5, 1.1],
        [0.5, 0.9],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect for vertical slit."
    assert np.allclose(result, expected), "Output values are incorrect for vertical slit."


def test_create_slit_wound_small_distance():
    """
    Test that the function handles a small slit_dist_from_mid correctly.
    """
    # Define input
    slit = np.array([
        [0.5, 0.5],
        [1.0, 0.5],
    ])
    slit_dir = 'x'
    slit_dist_from_mid = 0.01

    # Call the function
    result = mg.create_slit_wound(slit, slit_dir, slit_dist_from_mid)

    # Expected output
    expected = np.array([
        [0.49, 0.5],
        [0.51, 0.5],
        [1.01, 0.5],
        [0.99, 0.5],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect for small slit_dist_from_mid."
    assert np.allclose(result, expected), "Output values are incorrect for small slit_dist_from_mid."


def test_create_slit_wound_large_distance():
    """
    Test that the function handles a large slit_dist_from_mid correctly.
    """
    # Define input
    slit = np.array([
        [0.5, 0.5],
        [1.0, 0.5],
    ])
    slit_dir = 'x'
    slit_dist_from_mid = 0.5

    # Call the function
    result = mg.create_slit_wound(slit, slit_dir, slit_dist_from_mid)

    # Expected output
    expected = np.array([
        [0.0, 0.5],
        [1.0, 0.5],
        [1.5, 0.5],
        [0.5, 0.5],
    ])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect for large slit_dist_from_mid."
    assert np.allclose(result, expected), "Output values are incorrect for large slit_dist_from_mid."


def test_create_slit_wound_invalid_slit_shape():
    """
    Test that the function raises an error for invalid slit shape.
    """
    # Define invalid slit
    invalid_slit = np.array([
        [0.5, 0.5, 0.5],  # Incorrect shape
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="slit must be a 2D array with shape"):
        mg.create_slit_wound(invalid_slit, slit_dir='x', slit_dist_from_mid=0.1)


def test_create_slit_wound_invalid_slit_dir():
    """
    Test that the function raises an error for invalid slit_dir.
    """
    # Define valid slit
    slit = np.array([
        [0.5, 0.5],
        [1.0, 0.5],
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="slit_dir must be either 'x' or 'y'"):
        mg.create_slit_wound(slit, slit_dir='z', slit_dist_from_mid=0.1)


def test_create_slit_wound_invalid_slit_dist_from_mid():
    """
    Test that the function raises an error for invalid slit_dist_from_mid.
    """
    # Define valid slit
    slit = np.array([
        [0.5, 0.5],
        [1.0, 0.5],
    ])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="slit_dist_from_mid must be a positive number."):
        mg.create_slit_wound(slit, slit_dir='x', slit_dist_from_mid=-0.1)


def test_lines_on_posts_with_intersections():
    """
    Test that the function correctly identifies lines intersecting with posts.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Intersects with the first post
        [2, 2, 3, 3],  # Does not intersect with any post
        [0.9, 0.9, 1.1, 1.1],  # Intersects with the second post
    ])

    # Define posts
    posts_cent = np.array([
        [0.5, 0.5],  # First post center
        [1.0, 1.0],  # Second post center
    ])
    posts_rad = 0.5

    # Call the function
    result = mg.lines_on_posts(lines, posts_cent, posts_rad)

    # Expected output: indices of lines intersecting with posts
    expected = np.array([0, 2])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect."
    assert np.allclose(result, expected), "Output values are incorrect."


def test_lines_on_posts_no_intersections():
    """
    Test that the function returns an empty array when no lines intersect with posts.
    """
    # Define lines
    lines = np.array([
        [2, 2, 3, 3],  # Completely outside any post
        [4, 4, 5, 5],  # Completely outside any post
    ])

    # Define posts
    posts_cent = np.array([
        [0.5, 0.5],  # Post center
    ])
    posts_rad = 0.5

    # Call the function
    result = mg.lines_on_posts(lines, posts_cent, posts_rad)

    # Check the result
    assert result.shape == (0,), "Output should be an empty array when no lines intersect."


def test_lines_on_posts_tangential_lines():
    """
    Test that the function correctly identifies lines tangential to posts.
    """
    # Define lines
    lines = np.array([
        [0, 0.5, 1, 0.5],  # Tangential to the post
    ])

    # Define posts
    posts_cent = np.array([
        [0.5, 0.5],  # Post center
    ])
    posts_rad = 0.5

    # Call the function
    result = mg.lines_on_posts(lines, posts_cent, posts_rad)

    # Expected output: indices of tangential lines
    expected = np.array([0])

    # Check the result
    assert result.shape == expected.shape, "Output shape is incorrect for tangential lines."
    assert np.allclose(result, expected), "Output values are incorrect for tangential lines."


def test_lines_on_posts_invalid_lines_shape():
    """
    Test that the function raises an error for invalid lines shape.
    """
    # Define invalid lines
    invalid_lines = np.array([
        [0, 0, 1],  # Incorrect shape
    ])

    # Define valid posts
    posts_cent = np.array([
        [0.5, 0.5],
    ])
    posts_rad = 0.5

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines must be a 2D array with shape"):
        mg.lines_on_posts(invalid_lines, posts_cent, posts_rad)


def test_lines_on_posts_invalid_posts_shape():
    """
    Test that the function raises an error for invalid posts_cent shape.
    """
    # Define valid lines
    lines = np.array([
        [0, 0, 1, 1],
    ])

    # Define invalid posts
    invalid_posts_cent = np.array([
        [0.5],  # Incorrect shape
    ])
    posts_rad = 0.5

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="posts_cent must be a 2D array with shape"):
        mg.lines_on_posts(lines, invalid_posts_cent, posts_rad)


def test_lines_on_posts_invalid_posts_rad():
    """
    Test that the function raises an error for invalid posts_rad.
    """
    # Define valid lines and posts
    lines = np.array([
        [0, 0, 1, 1],
    ])
    posts_cent = np.array([
        [0.5, 0.5],
    ])

    # Call the function with invalid posts_rad
    with pytest.raises(ValueError, match="posts_rad must be a positive number."):
        mg.lines_on_posts(lines, posts_cent, posts_rad=-0.5)


def test_lines_on_posts_invalid_tol():
    """
    Test that the function raises an error for invalid tol.
    """
    # Define valid lines and posts
    lines = np.array([
        [0, 0, 1, 1],
    ])
    posts_cent = np.array([
        [0.5, 0.5],
    ])
    posts_rad = 0.5

    # Call the function with invalid tol
    with pytest.raises(ValueError, match="tol must be a non-negative number."):
        mg.lines_on_posts(lines, posts_cent, posts_rad, tol=-0.1)


def test_dict_move_items_to_another_key_valid_move():
    """
    Test moving items from one key to another.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
        4: {5},
        6: set(),
    }

    # Call the function
    result = mg.dict_move_items_to_another_key(cur_dict, key_to_remove=1, key_to_add=4)

    # Expected output
    expected = {
        4: {1,2,3,5},
        6: set(),
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect."


def test_dict_move_items_to_another_key_new_key():
    """
    Test moving items to a new key that does not exist.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
        4: {5},
    }

    # Call the function
    result = mg.dict_move_items_to_another_key(cur_dict, key_to_remove=1, key_to_add=7)

    # Expected output
    expected = {
        4: {5},
        7: {1, 2, 3},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect when moving to a new key."


def test_dict_move_items_to_another_key_same_key():
    """
    Test moving items when key_to_remove equals key_to_add.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
        4: {5},
    }

    # Call the function
    result = mg.dict_move_items_to_another_key(cur_dict, key_to_remove=1, key_to_add=1)

    # Expected output: no changes
    expected = {
        1: {2, 3},
        4: {5},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect when key_to_remove equals key_to_add."


def test_dict_move_items_to_another_key_empty_dict():
    """
    Test moving items in an empty dictionary.
    """
    # Define input dictionary
    cur_dict = {}

    # Call the function and check for KeyError
    with pytest.raises(KeyError, match="key_to_remove"):
        mg.dict_move_items_to_another_key(cur_dict, key_to_remove=1, key_to_add=4)


def test_dict_move_items_to_another_key_invalid_dict():
    """
    Test that the function raises an error when cur_dict is not a dictionary.
    """
    # Define invalid input
    cur_dict = "not_a_dict"

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="cur_dict must be a dictionary."):
        mg.dict_move_items_to_another_key(cur_dict, key_to_remove=1, key_to_add=4)


def test_dict_move_items_to_another_key_invalid_keys():
    """
    Test that the function raises an error when keys are not integers.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
        4: {5},
    }

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="key_to_remove and key_to_add must be integers."):
        mg.dict_move_items_to_another_key(cur_dict, key_to_remove="1", key_to_add=4)


def test_dict_move_items_to_another_key_key_not_found():
    """
    Test that the function raises an error when key_to_remove does not exist.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
        4: {5},
    }

    # Call the function and check for KeyError
    with pytest.raises(KeyError, match="key_to_remove"):
        mg.dict_move_items_to_another_key(cur_dict, key_to_remove=7, key_to_add=4)


def test_dict_add_item_to_key_add_new_key():
    """
    Test adding a new key to an existing set.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
        4: {5},
    }

    # Call the function
    result = mg.dict_add_item_to_key(cur_dict, key_new=6, key_add=4)

    # Expected output
    expected = {
        1: {2, 3},
        4: {5, 6},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect when adding a new key."


def test_dict_add_item_to_key_add_to_new_set():
    """
    Test adding a key to a new set.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
    }

    # Call the function
    result = mg.dict_add_item_to_key(cur_dict, key_new=4, key_add=5)

    # Expected output
    expected = {
        1: {2, 3},
        5: {4},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect when adding to a new set."


def test_dict_add_item_to_key_key_already_in_set():
    """
    Test adding a key that is already in the target set.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
        4: {5},
    }

    # Call the function
    result = mg.dict_add_item_to_key(cur_dict, key_new=5, key_add=4)

    # Expected output: no changes
    expected = {
        1: {2, 3},
        4: {5},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect when key is already in the target set."


def test_dict_add_item_to_key_same_key():
    """
    Test adding a key that is the same as the target key.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
        4: {5},
    }

    # Call the function
    result = mg.dict_add_item_to_key(cur_dict, key_new=4, key_add=4)

    # Expected output: no changes
    expected = {
        1: {2, 3},
        4: {5},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect when key_new equals key_add."


def test_dict_add_item_to_key_empty_dict():
    """
    Test adding a key to an empty dictionary.
    """
    # Define input dictionary
    cur_dict = {}

    # Call the function
    result = mg.dict_add_item_to_key(cur_dict, key_new=1, key_add=2)

    # Expected output
    expected = {
        2: {1},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect when adding to an empty dictionary."


def test_dict_add_item_to_key_invalid_dict():
    """
    Test that the function raises an error when cur_dict is not a dictionary.
    """
    # Define invalid input
    cur_dict = "not_a_dict"

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="cur_dict must be a dictionary."):
        mg.dict_add_item_to_key(cur_dict, key_new=1, key_add=2)


def test_dict_add_item_to_key_invalid_keys():
    """
    Test that the function raises an error when keys are not integers.
    """
    # Define input dictionary
    cur_dict = {
        1: {2, 3},
    }

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="key_new and key_add must be integers."):
        mg.dict_add_item_to_key(cur_dict, key_new="1", key_add=2)


def test_organize_dict_direct_keys():
    """
    Test organizing a dictionary where keys are directly in lines_on_posts_ind.
    """
    # Define input dictionary and lines_on_posts_ind
    cur_dict = {
        1: {2, 3},
        4: {5},
        6: {7},
    }
    lines_on_posts_ind = np.array([1, 4])

    # Call the function
    result = mg.organize_dict(cur_dict, lines_on_posts_ind)

    # Expected output
    expected = {
        1: {2, 3},
        4: {5},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect for direct keys."


def test_organize_dict_indirect_keys():
    """
    Test organizing a dictionary where keys are indirectly related to lines_on_posts_ind.
    """
    # Define input dictionary and lines_on_posts_ind
    cur_dict = {
        1: {2, 3},
        4: {5},
        6: {1, 7},
    }
    lines_on_posts_ind = np.array([1, 5])

    # Call the function
    result = mg.organize_dict(cur_dict, lines_on_posts_ind)

    # Expected output
    expected = {
        1: {2, 3, 6, 7},
        5: {4},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect for indirect keys."


def test_organize_dict_empty_dict():
    """
    Test organizing an empty dictionary.
    """
    # Define input dictionary and lines_on_posts_ind
    cur_dict = {}
    lines_on_posts_ind = np.array([1, 4])

    # Call the function
    result = mg.organize_dict(cur_dict, lines_on_posts_ind)

    # Expected output
    expected = {
        1: set(),
        4: set(),
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect for an empty input dictionary."


def test_organize_dict_empty_lines_on_posts_ind():
    """
    Test organizing a dictionary with an empty lines_on_posts_ind.
    """
    # Define input dictionary and lines_on_posts_ind
    cur_dict = {
        1: {2, 3},
        4: {5},
    }
    lines_on_posts_ind = np.array([])

    # Call the function
    result = mg.organize_dict(cur_dict, lines_on_posts_ind)

    # Expected output
    expected = {}

    # Check the result
    assert result == expected, "Output dictionary is incorrect for an empty lines_on_posts_ind."


def test_organize_dict_no_keys_in_lines_on_posts_ind():
    """
    Test organizing a dictionary where no keys are in lines_on_posts_ind.
    """
    # Define input dictionary and lines_on_posts_ind
    cur_dict = {
        1: {2, 3},
        4: {5},
    }
    lines_on_posts_ind = np.array([6, 7])

    # Call the function
    result = mg.organize_dict(cur_dict, lines_on_posts_ind)

    # Expected output
    expected = {
        6: set(),
        7: set(),
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect when no keys are in lines_on_posts_ind."


def test_organize_dict_all_keys_in_lines_on_posts_ind():
    """
    Test organizing a dictionary where all keys are in lines_on_posts_ind.
    """
    # Define input dictionary and lines_on_posts_ind
    cur_dict = {
        1: {2, 3},
        4: {5},
    }
    lines_on_posts_ind = np.array([1, 4])

    # Call the function
    result = mg.organize_dict(cur_dict, lines_on_posts_ind)

    # Expected output
    expected = {
        1: {2, 3},
        4: {5},
    }

    # Check the result
    assert result == expected, "Output dictionary is incorrect when all keys are in lines_on_posts_ind."


def test_organize_dict_invalid_cur_dict():
    """
    Test that the function raises an error when cur_dict is not a dictionary.
    """
    # Define invalid input
    cur_dict = "not_a_dict"
    lines_on_posts_ind = np.array([1, 4])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="cur_dict must be a dictionary."):
        mg.organize_dict(cur_dict, lines_on_posts_ind)


def test_organize_dict_invalid_lines_on_posts_ind():
    """
    Test that the function raises an error when lines_on_posts_ind is not a NumPy array.
    """
    # Define input dictionary and invalid lines_on_posts_ind
    cur_dict = {
        1: {2, 3},
        4: {5},
    }
    lines_on_posts_ind = [1, 4]  # Not a NumPy array

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines_on_posts_ind must be a NumPy array."):
        mg.organize_dict(cur_dict, lines_on_posts_ind)


def test_lines_intersect_lines_on_posts_with_intersections():
    """
    Test that the function correctly identifies lines intersecting with lines on posts.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0
        [1, 1, 2, 2],  # Line 1 (intersects with Line 0)
        [2, 2, 3, 3],  # Line 2 (intersects with Line 1)
        [3.5, 3.5, 4, 4],  # Line 3 (does not intersect with any line on posts)
    ])
    lines_on_posts_ind = [0]  # Line 0 is directly on the post

    # Call the function
    result = mg.lines_intersect_lines_on_posts(lines, lines_on_posts_ind)

    # Expected output
    expected = np.array([1, 2])

    # Check the result
    assert np.array_equal(np.sort(result), np.sort(expected)), "Output is incorrect for intersecting lines."


def test_lines_intersect_lines_on_posts_no_intersections():
    """
    Test that the function returns an empty array when no lines intersect with lines on posts.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0
        [2, 2, 3, 3],  # Line 1
        [4, 4, 5, 5],  # Line 2
    ])
    lines_on_posts_ind = [0]  # Line 0 is directly on the post

    # Call the function
    result = mg.lines_intersect_lines_on_posts(lines, lines_on_posts_ind)

    # Check the result
    assert result.size == 0, "Output should be an empty array when no intersections exist."


def test_lines_intersect_lines_on_posts_tangential_lines():
    """
    Test that the function correctly identifies lines tangential to lines on posts.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0
        [1, 1, 2, 2],  # Line 1 (tangential to Line 0)
    ])
    lines_on_posts_ind = [0]  # Line 0 is directly on the post

    # Call the function
    result = mg.lines_intersect_lines_on_posts(lines, lines_on_posts_ind)

    # Expected output
    expected = np.array([1])

    # Check the result
    assert np.array_equal(result, expected), "Output is incorrect for tangential lines."


def test_lines_intersect_lines_on_posts_empty_lines_on_posts_ind():
    """
    Test that the function returns an empty array when lines_on_posts_ind is empty.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0
        [1, 1, 2, 2],  # Line 1
    ])
    lines_on_posts_ind = []  # No lines are directly on posts

    # Call the function
    result = mg.lines_intersect_lines_on_posts(lines, lines_on_posts_ind)

    # Check the result
    assert result.size == 0, "Output should be an empty array when lines_on_posts_ind is empty."


def test_lines_intersect_lines_on_posts_all_lines_intersect():
    """
    Test that the function correctly identifies all lines when all intersect with lines on posts.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0
        [1, 1, 2, 2],  # Line 1 (intersects with Line 0)
        [2, 2, 3, 3],  # Line 2 (intersects with Line 1)
    ])
    lines_on_posts_ind = [0]  # Line 0 is directly on the post

    # Call the function
    result = mg.lines_intersect_lines_on_posts(lines, lines_on_posts_ind)

    # Expected output
    expected = np.array([1, 2])

    # Check the result
    assert np.array_equal(np.sort(result), np.sort(expected)), "Output is incorrect when all lines intersect."


def test_lines_intersect_lines_on_posts_invalid_lines():
    """
    Test that the function raises an error when lines is not a 2D array with shape (N, 4).
    """
    # Define invalid lines
    lines = np.array([
        [0, 0, 1],  # Incorrect shape
    ])
    lines_on_posts_ind = [0]

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines must be a 2D array with shape"):
        mg.lines_intersect_lines_on_posts(lines, lines_on_posts_ind)


def test_lines_intersect_lines_on_posts_invalid_lines_on_posts_ind():
    """
    Test that the function raises an error when lines_on_posts_ind is not a list or NumPy array.
    """
    # Define valid lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0
        [1, 1, 2, 2],  # Line 1
    ])
    lines_on_posts_ind = "not_a_list"  # Invalid type

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="lines_on_posts_ind must be a list or NumPy array."):
        mg.lines_intersect_lines_on_posts(lines, lines_on_posts_ind)


def test_remove_floating_lines_valid_input():
    """
    Test removing floating lines with valid input.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0 (in contact with post)
        [1, 1, 2, 2],  # Line 1 (intersects with Line 0)
        [3, 3, 4, 4],  # Line 2 (floating line)
    ])
    posts_cent = np.array([
        [0.5, 0.5],  # Post center
    ])
    posts_rad = 0.5

    # Call the function
    result = mg.remove_floating_lines(lines, posts_cent, posts_rad)

    # Expected output
    expected = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
    ])

    # Check the result
    assert np.array_equal(result, expected), "Output is incorrect for valid input."


def test_remove_floating_lines_no_lines_on_posts():
    """
    Test removing floating lines when no lines are in contact with posts.
    """
    # Define lines
    lines = np.array([
        [2, 2, 3, 3],  # Line 0
        [4, 4, 5, 5],  # Line 1
    ])
    posts_cent = np.array([
        [0.5, 0.5],  # Post center
    ])
    posts_rad = 0.5

    # Check the result
    with pytest.raises(ValueError, match="Ill-pose boundary conditions, tissue fiber network is not attached to pillars."):
        mg.remove_floating_lines(lines, posts_cent, posts_rad)


def test_remove_floating_lines_no_intersecting_lines():
    """
    Test removing floating lines when no lines intersect with lines on posts.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0 (in contact with post)
        [2, 2, 3, 3],  # Line 1 (does not intersect with Line 0)
    ])
    posts_cent = np.array([
        [0.5, 0.5],  # Post center
    ])
    posts_rad = 0.5

    # Call the function
    result = mg.remove_floating_lines(lines, posts_cent, posts_rad)

    # Expected output: only Line 0 remains
    expected = np.array([
        [0, 0, 1, 1],
    ])

    # Check the result
    assert np.array_equal(result, expected), "Output is incorrect when no lines intersect with lines on posts."


def test_remove_floating_lines_all_lines_intersect():
    """
    Test removing floating lines when all lines intersect with lines on posts.
    """
    # Define lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0 (in contact with post)
        [1, 1, 2, 2],  # Line 1 (intersects with Line 0)
        [2, 2, 3, 3],  # Line 2 (intersects with Line 1)
    ])
    posts_cent = np.array([
        [0.5, 0.5],  # Post center
    ])
    posts_rad = 0.5

    # Call the function
    result = mg.remove_floating_lines(lines, posts_cent, posts_rad)

    # Expected output: all lines remain
    expected = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 3, 3],
    ])

    # Check the result
    assert np.array_equal(result, expected), "Output is incorrect when all lines intersect."


def test_remove_floating_lines_invalid_posts_cent():
    """
    Test that the function raises an error when posts_cent is not a 2D array with shape (M, 2).
    """
    # Define valid lines
    lines = np.array([
        [0, 0, 1, 1],  # Line 0
    ])
    posts_cent = np.array([
        [0.5],  # Incorrect shape
    ])
    posts_rad = 0.5

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="posts_cent must be a 2D array with shape"):
        mg.remove_floating_lines(lines, posts_cent, posts_rad)


def test_remove_floating_lines_invalid_posts_rad():
    """
    Test that the function raises an error when posts_rad is zero or negative.
    """
    # Define valid lines and posts_cent
    lines = np.array([
        [0, 0, 1, 1],  # Line 0
    ])
    posts_cent = np.array([
        [0.5, 0.5],  # Post center
    ])
    posts_rad = -0.5  # Invalid radius

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="posts_rad must be a positive number."):
        mg.remove_floating_lines(lines, posts_cent, posts_rad)


def test_assign_radius_to_fibers_normal_distribution():
    """
    Test assigning radii using a normal distribution.
    """
    # Define input
    lines = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 3, 3],
    ])
    mean_radii = 0.05
    sd_radii = 0.01

    # Call the function
    radii = mg.assign_radius_to_fibers(lines, mean_radii, sd_radii, radii_dist_type='normal')

    # Check the result
    assert radii.shape == (3,), "Output radii array has incorrect shape."
    assert np.all(radii > 0), "Radii should be positive."


def test_assign_radius_to_fibers_uniform_distribution():
    """
    Test assigning radii using a uniform distribution.
    """
    # Define input
    lines = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 3, 3],
    ])
    mean_radii = 0.05
    sd_radii = 0.01

    # Call the function
    radii = mg.assign_radius_to_fibers(lines, mean_radii, sd_radii, radii_dist_type='uniform')

    # Check the result
    assert radii.shape == (3,), "Output radii array has incorrect shape."
    assert np.all(radii >= mean_radii - sd_radii), "Radii should not be less than mean_radii - sd_radii."
    assert np.all(radii <= mean_radii + sd_radii), "Radii should not be greater than mean_radii + sd_radii."


def test_assign_radius_to_fibers_constant_distribution():
    """
    Test assigning radii using a constant distribution.
    """
    # Define input
    lines = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 3, 3],
    ])
    mean_radii = 0.05
    sd_radii = 0.01

    # Call the function
    radii = mg.assign_radius_to_fibers(lines, mean_radii, sd_radii, radii_dist_type='constant')

    # Check the result
    assert radii.shape == (3,), "Output radii array has incorrect shape."
    assert np.all(radii == mean_radii), "All radii should be equal to mean_radii."


def test_assign_radius_to_fibers_no_fibers():
    """
    Test assigning radii when there are no fibers (empty lines array).
    """
    # Define input
    lines = np.empty((0, 4))
    mean_radii = 0.05
    sd_radii = 0.01

    # Call the function
    radii = mg.assign_radius_to_fibers(lines, mean_radii, sd_radii, radii_dist_type='normal')

    # Check the result
    assert radii.size == 0, "Output should be an empty array when there are no fibers."


def test_assign_radius_to_fibers_zero_sd():
    """
    Test assigning radii with zero standard deviation.
    """
    # Define input
    lines = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 3, 3],
    ])
    mean_radii = 0.05
    sd_radii = 0.0

    # Call the function
    radii = mg.assign_radius_to_fibers(lines, mean_radii, sd_radii, radii_dist_type='normal')

    # Check the result
    assert radii.shape == (3,), "Output radii array has incorrect shape."
    assert np.all(radii == mean_radii), "All radii should be equal to mean_radii when sd_radii is zero."


def test_assign_radius_to_fibers_invalid_mean_radii():
    """
    Test that the function raises an error when mean_radii is not positive.
    """
    # Define input
    lines = np.array([
        [0, 0, 1, 1],
    ])
    mean_radii = -0.05  # Invalid mean_radii
    sd_radii = 0.01

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="mean_radii must be a positive number."):
        mg.assign_radius_to_fibers(lines, mean_radii, sd_radii, radii_dist_type='normal')


def test_assign_radius_to_fibers_invalid_sd_radii():
    """
    Test that the function raises an error when sd_radii is negative.
    """
    # Define input
    lines = np.array([
        [0, 0, 1, 1],
    ])
    mean_radii = 0.05
    sd_radii = -0.01  # Invalid sd_radii

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="sd_radii must be a non-negative number."):
        mg.assign_radius_to_fibers(lines, mean_radii, sd_radii, radii_dist_type='normal')


def test_assign_radius_to_fibers_invalid_radii_dist_type():
    """
    Test that the function raises an error when radii_dist_type is invalid.
    """
    # Define input
    lines = np.array([
        [0, 0, 1, 1],
    ])
    mean_radii = 0.05
    sd_radii = 0.01

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="radii_dist_type must be one of 'normal', 'uniform', or 'constant'."):
        mg.assign_radius_to_fibers(lines, mean_radii, sd_radii, radii_dist_type='invalid_type')


def test_assign_small_fib_to_original_exact_match():
    """
    Test assigning indices when split lines match exactly with original lines.
    """
    # Define input
    og_lines = np.array([
        [0, 0, 1, 1],  # Original line 0
        [1, 1, 2, 2],  # Original line 1
    ])
    split_lines = np.array([
        [0, 0, 1, 1],  # Split line 0 matches original line 0
        [1, 1, 2, 2],  # Split line 1 matches original line 1
    ])
    old_ind = np.array([10, 20])  # Indices of original lines

    # Call the function
    result = mg.assign_small_fib_to_original(og_lines, split_lines, old_ind)

    # Expected output
    expected = np.array([10, 20])

    # Check the result
    assert np.array_equal(result, expected), "Output is incorrect for exact match."


def test_assign_small_fib_to_original_partial_overlap():
    """
    Test assigning indices when split lines partially overlap with original lines.
    """
    # Define input
    og_lines = np.array([
        [0, 0, 1, 1],  # Original line 0
        [1, 1, 2, 2],  # Original line 1
    ])
    split_lines = np.array([
        [0, 0, 0.5, 0.5],  # Split line 0 overlaps with original line 0
        [1.5, 1.5, 2, 2],  # Split line 1 overlaps with original line 1
    ])
    old_ind = np.array([10, 20])  # Indices of original lines

    # Call the function
    result = mg.assign_small_fib_to_original(og_lines, split_lines, old_ind)

    # Expected output
    expected = np.array([10, 20])

    # Check the result
    assert np.array_equal(result, expected), "Output is incorrect for partial overlap."


def test_assign_small_fib_to_original_no_split_lines():
    """
    Test assigning indices when there are no split lines.
    """
    # Define input
    og_lines = np.array([
        [0, 0, 1, 1],  # Original line 0
        [1, 1, 2, 2],  # Original line 1
    ])
    split_lines = np.empty((0, 4))  # No split lines
    old_ind = np.array([10, 20])  # Indices of original lines

    # Call the function
    result = mg.assign_small_fib_to_original(og_lines, split_lines, old_ind)

    # Check the result
    assert result.size == 0, "Output should be an empty array when there are no split lines."


def test_assign_small_fib_to_original_no_original_lines():
    """
    Test assigning indices when there are no original lines.
    """
    # Define input
    og_lines = np.empty((0, 4))  # No original lines
    split_lines = np.array([
        [0, 0, 0.5, 0.5],  # Split line 0
    ])
    old_ind = np.array([])  # No indices for original lines

    # Call the function
    result = mg.assign_small_fib_to_original(og_lines, split_lines, old_ind)

    # Check the result
    assert result.size == 0, "Output should be an empty array when there are no original lines."


def test_assign_small_fib_to_original_zero_tolerance():
    """
    Test assigning indices with zero tolerance.
    """
    # Define input
    og_lines = np.array([
        [0, 0, 1, 1],  # Original line 0
    ])
    split_lines = np.array([
        [0, 0, 0.5, 0.5],  # Split line 0
    ])
    old_ind = np.array([10])  # Indices of original lines

    # Call the function
    result = mg.assign_small_fib_to_original(og_lines, split_lines, old_ind, line_tol=0)

    # Check the result
    assert result.size == 0, "Output should be an empty array with zero tolerance."


def test_assign_small_fib_to_original_invalid_og_lines():
    """
    Test that the function raises an error when og_lines is not a 2D array with shape (N, 4).
    """
    # Define invalid og_lines
    og_lines = np.array([
        [0, 0, 1],  # Incorrect shape
    ])
    split_lines = np.array([
        [0, 0, 0.5, 0.5],
    ])
    old_ind = np.array([10])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="og_lines must be a 2D array with shape"):
        mg.assign_small_fib_to_original(og_lines, split_lines, old_ind)


def test_assign_small_fib_to_original_invalid_split_lines():
    """
    Test that the function raises an error when split_lines is not a 2D array with shape (M, 4).
    """
    # Define invalid split_lines
    og_lines = np.array([
        [0, 0, 1, 1],
    ])
    split_lines = np.array([
        [0, 0, 0.5],  # Incorrect shape
    ])
    old_ind = np.array([10])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="split_lines must be a 2D array with shape"):
        mg.assign_small_fib_to_original(og_lines, split_lines, old_ind)


def test_assign_small_fib_to_original_invalid_old_ind():
    """
    Test that the function raises an error when old_ind is not a 1D array or does not match the length of og_lines.
    """
    # Define invalid old_ind
    og_lines = np.array([
        [0, 0, 1, 1],
    ])
    split_lines = np.array([
        [0, 0, 0.5, 0.5],
    ])
    old_ind = np.array([[10]])  # Incorrect shape

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="old_ind must be a 1D array with the same length as og_lines"):
        mg.assign_small_fib_to_original(og_lines, split_lines, old_ind)


def test_assign_small_fib_to_original_negative_tolerance():
    """
    Test that the function raises an error when line_tol is negative.
    """
    # Define input
    og_lines = np.array([
        [0, 0, 1, 1],
    ])
    split_lines = np.array([
        [0, 0, 0.5, 0.5],
    ])
    old_ind = np.array([10])

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="line_tol must be a non-negative number."):
        mg.assign_small_fib_to_original(og_lines, split_lines, old_ind, line_tol=-0.1)


def test_generate_lines_for_circle_small_circle():
    """
    Test generating a circle with a small number of lines.
    """
    # Define input
    center = np.array([0, 0])
    radius = 1
    num_lines = 4

    # Call the function
    result = mg.generate_lines_for_circle(center, radius, num_lines)

    # Expected output: 4 lines forming a square-like approximation of a circle
    expected = np.array([
        [-1,0,0,-1],
        [-1,0,0,1],
        [0,-1,1,0],
        [0,1,1,0],
    ])

    # Check the result
    assert result.shape == (4, 4), "Output shape is incorrect."
    assert np.allclose(result, expected, atol=1e-6), "Output lines are incorrect."


def test_generate_lines_for_circle_large_circle():
    """
    Test generating a circle with a large number of lines.
    """
    # Define input
    center = np.array([0, 0])
    radius = 1
    num_lines = 100

    # Call the function
    result = mg.generate_lines_for_circle(center, radius, num_lines)

    # Check the result
    assert result.shape == (100, 4), "Output shape is incorrect."
    assert np.allclose(result[:, :2], result[:, 2:], atol=1e-6) is False, "Start and end points should not match."


def test_generate_lines_for_circle_one_line():
    """
    Test generating a circle with only one line.
    """
    # Define input
    center = np.array([0, 0])
    radius = 1
    num_lines = 1

    # Call the function
    result = mg.generate_lines_for_circle(center, radius, num_lines)

    # Expected output: a single line forming a degenerate circle
    expected = np.array([[1, 0, 1, 0]])

    # Check the result
    assert result.shape == (1, 4), "Output shape is incorrect."
    assert np.allclose(result, expected, atol=1e-6), "Output lines are incorrect."


def test_generate_lines_for_circle_invalid_center():
    """
    Test that the function raises an error when center is not a 1D array with shape (2,).
    """
    # Define invalid center
    center = np.array([0, 0, 0])  # Incorrect shape
    radius = 1
    num_lines = 4

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="center must be a 1D array with shape"):
        mg.generate_lines_for_circle(center, radius, num_lines)


def test_generate_lines_for_circle_invalid_radius():
    """
    Test that the function raises an error when radius is zero or negative.
    """
    # Define invalid radius
    center = np.array([0, 0])
    radius = -1  # Invalid radius
    num_lines = 4

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="radius must be a positive number."):
        mg.generate_lines_for_circle(center, radius, num_lines)


def test_generate_lines_for_circle_invalid_num_lines():
    """
    Test that the function raises an error when num_lines is zero or negative.
    """
    # Define invalid num_lines
    center = np.array([0, 0])
    radius = 1
    num_lines = 0  # Invalid num_lines

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="num_lines must be a positive integer."):
        mg.generate_lines_for_circle(center, radius, num_lines)


def test_add_circular_posts_scalar_radius():
    """
    Test adding circular posts with a scalar radius.
    """
    # Define input
    fib_net = np.array([
        [0, 0, 1, 1],  # Fiber 1
        [1, 1, 2, 2],  # Fiber 2
    ])
    posts_pos = np.array([
        [0, 0],  # Post 1 center
        [2, 2],  # Post 2 center
    ])
    posts_radius = 0.5  # Scalar radius
    num_lines = 8

    # Call the function
    result = mg.add_circular_posts(fib_net, posts_pos, posts_radius, num_lines)

    # Check the result
    assert result.shape[0] == fib_net.shape[0] + posts_pos.shape[0] * num_lines, \
        "Output shape is incorrect."
    assert result.shape[1] == 4, "Output should have 4 columns."


def test_add_circular_posts_array_radius():
    """
    Test adding circular posts with an array of radii.
    """
    # Define input
    fib_net = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    posts_pos = np.array([
        [0, 0],  # Post 1 center
        [2, 2],  # Post 2 center
    ])
    posts_radius = np.array([0.5, 0.3])  # Array of radii
    num_lines = 8

    # Call the function
    result = mg.add_circular_posts(fib_net, posts_pos, posts_radius, num_lines)

    # Check the result
    assert result.shape[0] == fib_net.shape[0] + posts_pos.shape[0] * num_lines, \
        "Output shape is incorrect."
    assert result.shape[1] == 4, "Output should have 4 columns."


def test_add_circular_posts_empty_fib_net():
    """
    Test adding circular posts to an empty fiber network.
    """
    # Define input
    fib_net = np.empty((0, 4))  # Empty fiber network
    posts_pos = np.array([
        [0, 0],  # Post 1 center
    ])
    posts_radius = 0.5  # Scalar radius
    num_lines = 8

    # Call the function
    result = mg.add_circular_posts(fib_net, posts_pos, posts_radius, num_lines)

    # Check the result
    assert result.shape[0] == posts_pos.shape[0] * num_lines, \
        "Output shape is incorrect for an empty fiber network."
    assert result.shape[1] == 4, "Output should have 4 columns."


def test_add_circular_posts_single_post():
    """
    Test adding a single circular post.
    """
    # Define input
    fib_net = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    posts_pos = np.array([
        [0, 0],  # Single post center
    ])
    posts_radius = 0.5  # Scalar radius
    num_lines = 8

    # Call the function
    result = mg.add_circular_posts(fib_net, posts_pos, posts_radius, num_lines)

    # Check the result
    assert result.shape[0] == fib_net.shape[0] + num_lines, \
        "Output shape is incorrect for a single post."
    assert result.shape[1] == 4, "Output should have 4 columns."


def test_add_circular_posts_invalid_fib_net():
    """
    Test that the function raises an error when fib_net is not a 2D array with shape (N, 4).
    """
    # Define invalid fib_net
    fib_net = np.array([
        [0, 0, 1],  # Incorrect shape
    ])
    posts_pos = np.array([
        [0, 0],  # Post 1 center
    ])
    posts_radius = 0.5  # Scalar radius
    num_lines = 8

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="fib_net must be a 2D array with shape"):
        mg.add_circular_posts(fib_net, posts_pos, posts_radius, num_lines)


def test_add_circular_posts_invalid_posts_pos():
    """
    Test that the function raises an error when posts_pos is not a 2D array with shape (M, 2).
    """
    # Define invalid posts_pos
    fib_net = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    posts_pos = np.array([
        [0, 0, 0],  # Incorrect shape
    ])
    posts_radius = 0.5  # Scalar radius
    num_lines = 8

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="posts_pos must be a 2D array with shape"):
        mg.add_circular_posts(fib_net, posts_pos, posts_radius, num_lines)


def test_add_circular_posts_invalid_posts_radius():
    """
    Test that the function raises an error when posts_radius is not a scalar or a 1D array with the same length as posts_pos.
    """
    # Define invalid posts_radius
    fib_net = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    posts_pos = np.array([
        [0, 0],  # Post 1 center
        [2, 2],  # Post 2 center
    ])
    posts_radius = np.array([0.5])  # Incorrect length
    num_lines = 8

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="posts_radius must have the same length as posts_pos if it is an array"):
        mg.add_circular_posts(fib_net, posts_pos, posts_radius, num_lines)


def test_add_circular_posts_invalid_num_lines():
    """
    Test that the function raises an error when num_lines is zero or negative.
    """
    # Define invalid num_lines
    fib_net = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    posts_pos = np.array([
        [0, 0],  # Post 1 center
    ])
    posts_radius = 0.5  # Scalar radius
    num_lines = 0  # Invalid num_lines

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="num_lines must be a positive integer"):
        mg.add_circular_posts(fib_net, posts_pos, posts_radius, num_lines)


def test_assign_posts_ind_valid_input():
    """
    Test assigning indices with a valid fiber network and posts.
    """
    # Define input
    fib_net_old = np.array([
        [0, 0, 1, 1],  # Fiber 1
        [1, 1, 2, 2],  # Fiber 2
    ])
    fib_net_new = np.array([
        [0, 0, 1, 1],  # Fiber 1
        [1, 1, 2, 2],  # Fiber 2
        [0, 0, 0.5, 0.5],  # Post 1 line 1
        [0.5, 0.5, 1, 1],  # Post 1 line 2
    ])
    num_posts = 1
    num_lines_percir = 2

    # Call the function
    result = mg.assign_posts_ind(fib_net_old, fib_net_new, num_posts=num_posts, num_lines_percir=num_lines_percir)

    # Expected output
    expected = np.array([-1, -1, 10000, 10000])

    # Check the result
    assert np.array_equal(result, expected), "Output is incorrect for valid input."


def test_assign_posts_ind_empty_fiber_network():
    """
    Test assigning indices when the fiber network is empty.
    """
    # Define input
    fib_net_old = np.empty((0, 4))  # Empty fiber network
    fib_net_new = np.array([
        [0, 0, 0.5, 0.5],  # Post 1 line 1
        [0.5, 0.5, 1, 1],  # Post 1 line 2
    ])
    num_posts = 1
    num_lines_percir = 2

    # Call the function
    result = mg.assign_posts_ind(fib_net_old, fib_net_new, num_posts=num_posts, num_lines_percir=num_lines_percir)

    # Expected output
    expected = np.array([10000, 10000])

    # Check the result
    assert np.array_equal(result, expected), "Output is incorrect for an empty fiber network."


def test_assign_posts_ind_single_post():
    """
    Test assigning indices for a single post with a single line.
    """
    # Define input
    fib_net_old = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    fib_net_new = np.array([
        [0, 0, 1, 1],  # Fiber 1
        [0, 0, 0.5, 0.5],  # Post 1 line 1
    ])
    num_posts = 1
    num_lines_percir = 1

    # Call the function
    result = mg.assign_posts_ind(fib_net_old, fib_net_new, num_posts=num_posts, num_lines_percir=num_lines_percir)

    # Expected output
    expected = np.array([-1, 10000])

    # Check the result
    assert np.array_equal(result, expected), "Output is incorrect for a single post."


def test_assign_posts_ind_invalid_fib_net_old():
    """
    Test that the function raises an error when fib_net_old is not a 2D array with shape (N, 4).
    """
    # Define invalid fib_net_old
    fib_net_old = np.array([
        [0, 0, 1],  # Incorrect shape
    ])
    fib_net_new = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    num_posts = 1
    num_lines_percir = 1

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="fib_net_old must be a 2D array with shape"):
        mg.assign_posts_ind(fib_net_old, fib_net_new, num_posts=num_posts, num_lines_percir=num_lines_percir)


def test_assign_posts_ind_invalid_fib_net_new():
    """
    Test that the function raises an error when fib_net_new is not a 2D array with shape (M, 4).
    """
    # Define invalid fib_net_new
    fib_net_old = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    fib_net_new = np.array([
        [0, 0, 1],  # Incorrect shape
    ])
    num_posts = 1
    num_lines_percir = 1

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="fib_net_new must be a 2D array with shape"):
        mg.assign_posts_ind(fib_net_old, fib_net_new, num_posts=num_posts, num_lines_percir=num_lines_percir)


def test_assign_posts_ind_invalid_num_posts():
    """
    Test that the function raises an error when num_posts is zero or negative.
    """
    # Define invalid num_posts
    fib_net_old = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    fib_net_new = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    num_posts = -1  # Invalid num_posts
    num_lines_percir = 1

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="num_posts must be a positive integer"):
        mg.assign_posts_ind(fib_net_old, fib_net_new, num_posts=num_posts, num_lines_percir=num_lines_percir)


def test_assign_posts_ind_mismatch_post_segments():
    """
    Test that the function raises an error when the number of post segments does not match num_posts * num_lines_percir.
    """
    # Define input
    fib_net_old = np.array([
        [0, 0, 1, 1],  # Fiber 1
    ])
    fib_net_new = np.array([
        [0, 0, 1, 1],  # Fiber 1
        [0, 0, 0.5, 0.5],  # Post 1 line 1
    ])
    num_posts = 1
    num_lines_percir = 2  # Mismatch: only 1 line for the post

    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="Mismatch between the number of posts and the number of lines per post"):
        mg.assign_posts_ind(fib_net_old, fib_net_new, num_posts=num_posts, num_lines_percir=num_lines_percir)


def test_generate_fib_net_for_tissue_basic():
    """
    Test basic functionality of generate_fib_net_for_tissue with default parameters.
    """
    config = mg.FiberNetworkParams()
    result = mg.generate_fib_net_for_tissue(config)

    assert isinstance(result, mg.FiberNetworkResult), "Result should be an instance of FiberNetworkResult."
    assert result.fibers.shape[1] == 4, "Fibers should have 4 columns representing [x1, y1, x2, y2]."
    assert result.indices.shape[0] == result.fibers.shape[0], "Each fiber should have a corresponding index."
    # assert result.radii.shape[0] == result.fibers.shape[0], "Each fiber should have a corresponding radius."


def test_generate_fib_net_for_tissue_with_custom_fiber_length():
    """
    Test generate_fib_net_for_tissue with custom fiber length parameters.
    """
    config = mg.FiberNetworkParams()
    config.fiber_length["distribution"] = "normal"
    config.fiber_length["mean"] = 0.5
    config.fiber_length["std_dev"] = 0.1

    result = mg.generate_fib_net_for_tissue(config)

    assert result.fibers.shape[0] > 0, "Fibers should be generated."
    assert np.all(result.radii > 0), "All fiber radii should be positive."


def test_generate_fib_net_for_tissue_with_uniform_orientation():
    """
    Test generate_fib_net_for_tissue with uniform fiber orientation.
    """
    config = mg.FiberNetworkParams()
    config.fiber_orientation["distribution"] = "uniform"
    config.fiber_orientation["range"] = [0, 90]

    result = mg.generate_fib_net_for_tissue(config)

    assert result.fibers.shape[0] > 0, "Fibers should be generated."
    assert result.fibers.shape[1] == 4, "Fibers should have 4 columns representing [x1, y1, x2, y2]."


def test_generate_fib_net_for_tissue_with_wound_shape():
    """
    Test generate_fib_net_for_tissue with a wound shape.
    """
    config = mg.FiberNetworkParams()
    config.wound_shape = np.array([[0.4, 0.3], [0.6, 0.3], [0.6, 0.5], [0.4, 0.5]])

    result = mg.generate_fib_net_for_tissue(config)

    assert result.fibers.shape[0] > 0, "Fibers should be generated."
    assert np.all(result.fibers[:, :2] >= 0), "Fiber start points should be within bounds."
    assert np.all(result.fibers[:, 2:] >= 0), "Fiber end points should be within bounds."


def test_generate_fib_net_for_tissue_with_posts():
    """
    Test generate_fib_net_for_tissue with microposts.
    """
    config = mg.FiberNetworkParams()
    config.post_positions = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]])
    config.post_radius = 0.1

    result = mg.generate_fib_net_for_tissue(config)

    assert result.fibers.shape[0] > 0, "Fibers should be generated."
    assert np.all(result.radii > 0), "All fiber radii should be positive."


def test_generate_fib_net_for_tissue_with_invalid_config():
    """
    Test generate_fib_net_for_tissue with an invalid configuration.
    """
    config = mg.FiberNetworkParams()
    config.num_fibers = -10  # Invalid number of fibers

    with pytest.raises(ValueError, match="num_fibers must be a positive integer."):
        mg.generate_fib_net_for_tissue(config)


def test_generate_fib_net_for_tissue_with_no_fibers():
    """
    Test generate_fib_net_for_tissue with zero fibers.
    """
    config = mg.FiberNetworkParams()
    config.num_fibers = 0

    with pytest.raises(ValueError, match="num_fibers must be a positive integer."):
        mg.generate_fib_net_for_tissue(config)


def test_find_line_cells_basic():
    """
    Test the basic functionality of find_line_cells with a simple input.
    """
    lines = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 3, 3]
    ])
    expected_unique_points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    expected_line_cells = np.array([
        [0, 1],
        [1, 2],
        [2, 3]
    ])
    unique_points, line_cells = mg.find_line_cells(lines)
    assert np.array_equal(unique_points, expected_unique_points)
    assert np.array_equal(line_cells, expected_line_cells)


def test_find_line_cells_no_lines():
    """
    Test find_line_cells with an empty input.
    """
    lines = np.empty((0, 4))
    with pytest.raises(ValueError, match="The array of lines representing the fiber network is empty."):
        mg.find_line_cells(lines) 


def test_find_line_cells_single_line():
    """
    Test find_line_cells with a single line.
    """
    lines = np.array([
        [0, 0, 1, 1]
    ])
    expected_unique_points = np.array([
        [0, 0],
        [1, 1]
    ])
    expected_line_cells = np.array([
        [0, 1]
    ])
    unique_points, line_cells = mg.find_line_cells(lines)
    assert np.array_equal(unique_points, expected_unique_points)
    assert np.array_equal(line_cells, expected_line_cells)


def test_find_line_cells_complex_case():
    """
    Test find_line_cells with a more complex case.
    """
    lines = np.array([
        [0, 0, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 0, 0],  # Forms a triangle
        [3, 3, 4, 4]
    ])
    expected_unique_points = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ])
    expected_line_cells = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [3, 4]
    ])
    unique_points, line_cells = mg.find_line_cells(lines)
    assert np.array_equal(unique_points, expected_unique_points)
    assert np.array_equal(line_cells, expected_line_cells)


def test_match_char_len_no_split():
    """
    Test case where no lines need to be split because their lengths are within the characteristic length.
    """
    lines = np.array([[0, 0, 1, 0], [1, 1, 2, 2]])
    char_len = 2.0
    line_ind = np.array([0, 1])

    refined_lines, new_ind = mg.match_char_len(lines, char_len, line_ind)

    assert np.array_equal(refined_lines, lines), "Lines should remain unchanged."
    assert np.array_equal(new_ind, line_ind), "Indices should remain unchanged."


def test_match_char_len_split():
    """
    Test case where lines need to be split into smaller segments.
    """
    lines = np.array([[0, 0, 3, 0], [1, 1, 4, 1]])
    char_len = 1.0
    line_ind = np.array([0, 1])

    refined_lines, new_ind = mg.match_char_len(lines, char_len, line_ind)

    expected_lines = np.array([
        [0, 0, 1, 0],
        [1, 0, 2, 0],
        [2, 0, 3, 0],
        [1, 1, 2, 1],
        [2, 1, 3, 1],
        [3, 1, 4, 1]
    ])
    expected_indices = np.array([0, 0, 0, 1, 1, 1])

    assert np.array_equal(refined_lines, expected_lines), "Lines should be split correctly."
    assert np.array_equal(new_ind, expected_indices), "Indices should match the split lines."


def test_match_char_len_edge_case_zero_length():
    """
    Test case where a line has zero length.
    """
    lines = np.array([[0, 0, 0, 0]])
    char_len = 1.0
    line_ind = np.array([0])

    refined_lines, new_ind = mg.match_char_len(lines, char_len, line_ind)

    expected_lines = np.array([[0, 0, 0, 0]])
    expected_indices = np.array([0])

    assert np.array_equal(refined_lines, expected_lines), "Zero-length line should remain unchanged."
    assert np.array_equal(new_ind, expected_indices), "Indices should remain unchanged."


def test_match_char_len_large_char_len():
    """
    Test case where the characteristic length is very large, so no splitting occurs.
    """
    lines = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    char_len = 10.0
    line_ind = np.array([0, 1])

    refined_lines, new_ind = mg.match_char_len(lines, char_len, line_ind)

    assert np.array_equal(refined_lines, lines), "Lines should remain unchanged."
    assert np.array_equal(new_ind, line_ind), "Indices should remain unchanged."


def test_match_char_len_multiple_splits():
    """
    Test case where a line needs to be split into multiple segments.
    """
    lines = np.array([[0, 0, 5, 0]])
    char_len = 2.0
    line_ind = np.array([0])

    refined_lines, new_ind = mg.match_char_len(lines, char_len, line_ind)

    expected_lines = np.array([
        [0, 0, 1.666666666666666666667, 0],
        [1.666666666666666666667, 0, 3.333333333333333333, 0],
        [3.333333333333333333, 0, 5, 0]
    ])
    expected_indices = np.array([0, 0, 0])

    assert np.allclose(refined_lines, expected_lines), "Lines should be split into multiple segments."
    assert np.allclose(new_ind, expected_indices), "Indices should match the split lines."


def test_match_char_len_non_axis_aligned():
    """
    Test case where lines are not aligned with the axes.
    """
    lines = np.array([[0, 0, 3, 4]])
    char_len = 2.5
    line_ind = np.array([0])

    refined_lines, new_ind = mg.match_char_len(lines, char_len, line_ind)

    expected_lines = np.array([
        [0, 0, 1.5, 2.],
        [1.5, 2., 3, 4]
    ])
    expected_indices = np.array([0, 0])

    assert np.allclose(refined_lines, expected_lines, atol=1e-2), "Lines should be split correctly for non-axis-aligned cases."
    assert np.array_equal(new_ind, expected_indices), "Indices should match the split lines."


@pytest.fixture
def sample_fiber_network():
    """Fixture to provide a sample fiber network for testing."""
    fib_net = np.array([
        [0, 0, 1, 0],
        [1, 0, 2, 0],
        [2, 0, 3, 0]
    ])
    fib_ind = np.array([0, 1, 2])
    return fib_net, fib_ind


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture to provide a temporary directory for output files."""
    return tmp_path


def test_generate_fib_net_xdmf_meshio_valid_input(sample_fiber_network, temp_output_dir):
    """
    Test that the function generates an XDMF file correctly with valid input.
    """
    fib_net, fib_ind = sample_fiber_network
    f_name = "test_fiber_network"
    output_xdmf = str(temp_output_dir) + "/"
    characteristic_length = 1.5

    # Call the function
    mg.generate_fib_net_xdmf_meshio(
        fib_net=fib_net,
        f_name=f_name,
        output_xdmf=output_xdmf,
        fib_ind=fib_ind,
        characteristic_length=characteristic_length
    )

    # Check if the output file exists
    output_file = os.path.join(output_xdmf, f"{f_name}.xdmf")
    assert os.path.exists(output_file), "XDMF file was not created."

    # Check the contents of the file using meshio
    mesh = meshio.read(output_file)
    assert mesh.points.shape[1] == 2, "Points should be 2D."
    assert len(mesh.cells_dict["line"]) == len(fib_net), "Number of lines should match the input fiber network."
    assert "fibers" in mesh.cell_data, "Cell data should contain 'fibers'."
    assert len(mesh.cell_data["fibers"][0]) == len(fib_net), "Fiber indices should match the input."


def test_generate_fib_net_xdmf_meshio_no_fib_ind(sample_fiber_network, temp_output_dir):
    """
    Test that the function works correctly when `fib_ind` is not provided.
    """
    fib_net, _ = sample_fiber_network
    f_name = "test_fiber_network_no_fib_ind"
    output_xdmf = str(temp_output_dir) + "/"
    characteristic_length = 1.5

    # Call the function without `fib_ind`
    mg.generate_fib_net_xdmf_meshio(
        fib_net=fib_net,
        f_name=f_name,
        output_xdmf=output_xdmf,
        fib_ind=None,
        characteristic_length=characteristic_length
    )

    # Check if the output file exists
    output_file = os.path.join(output_xdmf, f"{f_name}.xdmf")
    assert os.path.exists(output_file), "XDMF file was not created."

    # Check the contents of the file using meshio
    mesh = meshio.read(output_file)
    assert mesh.points.shape[1] == 2, "Points should be 2D."
    assert len(mesh.cells_dict["line"]) == len(fib_net), "Number of lines should match the input fiber network."
    assert "fibers" in mesh.cell_data, "Cell data should contain 'fibers'."
    assert len(mesh.cell_data["fibers"][0]) == len(fib_net), "Fiber indices should match the input."


def test_generate_fib_net_xdmf_meshio_invalid_fib_net(temp_output_dir):
    """
    Test that the function raises a ValueError for an invalid `fib_net`.
    """
    fib_net = np.array([[0, 0, 1]])  # Invalid shape
    f_name = "invalid_fiber_network"
    output_xdmf = str(temp_output_dir) + "/"
    fib_ind = np.array([0])
    characteristic_length = 1.5

    with pytest.raises(ValueError, match="fib_net must be a 2D array with shape \\(N, 4\\)."):
        mg.generate_fib_net_xdmf_meshio(
            fib_net=fib_net,
            f_name=f_name,
            output_xdmf=output_xdmf,
            fib_ind=fib_ind,
            characteristic_length=characteristic_length
        )


def test_generate_fib_net_xdmf_meshio_invalid_characteristic_length(sample_fiber_network, temp_output_dir):
    """
    Test that the function raises a ValueError for an invalid `characteristic_length`.
    """
    fib_net, fib_ind = sample_fiber_network
    f_name = "invalid_characteristic_length"
    output_xdmf = str(temp_output_dir) + "/"
    characteristic_length = -1.0  # Invalid value

    with pytest.raises(ValueError, match="characteristic_length must be a positive number."):
        mg.generate_fib_net_xdmf_meshio(
            fib_net=fib_net,
            f_name=f_name,
            output_xdmf=output_xdmf,
            fib_ind=fib_ind,
            characteristic_length=characteristic_length
        )


def test_generate_fib_net_xdmf_meshio_invalid_fib_ind(sample_fiber_network, temp_output_dir):
    """
    Test that the function raises a ValueError for an invalid `fib_ind`.
    """
    fib_net, _ = sample_fiber_network
    f_name = "invalid_fib_ind"
    output_xdmf = str(temp_output_dir) + "/"
    fib_ind = np.array([0, 1])  # Mismatched length
    characteristic_length = 1.5

    with pytest.raises(ValueError, match="fib_ind must be a 1D array with the same length as fib_net."):
        mg.generate_fib_net_xdmf_meshio(
            fib_net=fib_net,
            f_name=f_name,
            output_xdmf=output_xdmf,
            fib_ind=fib_ind,
            characteristic_length=characteristic_length
        )
