import unittest
import numpy as np
from unittest.mock import MagicMock

from src.tracking_algorithm import ExtendedKalmanFilter, Target, ObjectTracker, run_tracking_objects, \
    calculate_mahalanobis_distance


class TestTrackingAlgorithm(unittest.TestCase):

    def setUp(self):
        """
        Set up the initial conditions for the unit tests.
        """
        # Set up a mock target with EKF
        self.target_id = 1
        self.initial_state = [0, 0, 0, 0]  # [x, y, velocity_x, velocity_y]
        self.ekf = ExtendedKalmanFilter(initial_state=self.initial_state)
        self.target = Target(obj_id=self.target_id, ekf=self.ekf)

        # Create a mock ObjectTracker
        self.tracker = ObjectTracker()

        # Add the target to the tracker
        self.tracker.tracked_objects[self.target_id] = self.target

    def test_mahalanobis_distance(self):
        """
        Test the Mahalanobis distance calculation.
        """
        prediction = np.array([0, 0])
        detection = np.array([1, 1])
        covariance_matrix = np.eye(4)  # Identity matrix for simplicity

        # Calculate the Mahalanobis distance
        dist = calculate_mahalanobis_distance(prediction, detection, covariance_matrix)

        # Check if the distance is correct
        expected_distance = np.linalg.norm(prediction - detection)  # Should match the Euclidean distance
        self.assertAlmostEqual(dist, expected_distance, places=3)

    def test_target_matching(self):
        """
        Test the Hungarian algorithm for target matching.
        """
        predictions = {self.target_id: self.target}  # The target to be matched
        detections = np.array([[0, 0]])  # One detection, at the same location

        # Perform the matching
        matches = self.tracker.matcher.match(predictions, detections)

        # Check if the correct matching occurs
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], self.target_id)  # The target should match with the detection
        self.assertEqual(matches[0][1], 0)  # Detection index should be 0

    def test_ekf_predict(self):
        """
        Test the Extended Kalman Filter prediction step.
        """
        initial_state = self.target.ekf.state
        predicted_state = self.target.ekf.predict()

        # The predicted state should be equal to the initial state in the absence of new measurements
        self.assertTrue(np.allclose(predicted_state, initial_state[:2]))  # Only check position (x, y)

    def test_ekf_update(self):
        """
        Test the Extended Kalman Filter update step.
        """
        new_measurement = [1, 1]  # New detection at (1, 1)

        # Perform the update
        self.target.ekf.update(new_measurement)

        # The state should have been updated to reflect the new measurement
        updated_state = self.target.ekf.state[:2]
        self.assertTrue(np.allclose(updated_state, new_measurement))

    def test_object_tracker_add_and_remove_targets(self):
        """
        Test the add and remove target functionality in ObjectTracker.
        """
        # Add new target
        new_detection = [2, 2, 1, 1]  # Detection at (2, 2) with size (1, 1)
        self.tracker.add_new_targets([new_detection])

        # Check if the target has been added
        self.assertEqual(len(self.tracker.tracked_objects), 2)

        # Simulate a missed detection for the target
        target = self.tracker.tracked_objects[2]
        target.miss_count = 3  # The target should now be removed after 3 misses

        # Remove lost targets
        self.tracker.remove_lost_targets()

        # Check if the target was removed
        self.assertEqual(len(self.tracker.tracked_objects), 1)

    def test_run_tracking_objects(self):
        """
        Test the overall tracking function (mocked for simplicity).
        """
        # Create mock ground truth data for a single frame
        gt_data = {
            'frame': [1],
            'x1': [0],
            'y1': [0],
            'w': [1],
            'h': [1]
        }

        # Mock the MetricsEvaluator to return dummy metrics
        evaluator_mock = MagicMock()
        evaluator_mock.evaluate_performance.return_value = {
            "frames": [1],
            "precision": [1.0],
            "recall": [1.0],
            "f1": [1.0],
            "mota": [1.0],
            "motp": [1.0]
        }

        # Replace the MetricsEvaluator with the mock
        self.tracker.evaluator = evaluator_mock

        # Run the tracking function
        metrics = run_tracking_objects(gt_data)

        # Verify that the mock was called
        evaluator_mock.evaluate_performance.assert_called_once()

        # Check the metrics
        self.assertEqual(metrics["precision"][0], 1.0)
        self.assertEqual(metrics["recall"][0], 1.0)
        self.assertEqual(metrics["f1"][0], 1.0)
        self.assertEqual(metrics["mota"][0], 1.0)
        self.assertEqual(metrics["motp"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
