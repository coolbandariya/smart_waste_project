import unittest

from person4_optimization import solve_vrp


class RouteOptimizationTests(unittest.TestCase):
    def test_empty_input_returns_no_routes(self):
        self.assertEqual(solve_vrp([]), {})

    def test_invalid_vehicle_count_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_vrp([], num_vehicles=0)

    def test_invalid_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            solve_vrp([{"latitude": 200, "longitude": 77.3, "waste_volume": 10}])

    def test_demand_over_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_vrp([{"latitude": 28.5, "longitude": 77.3, "waste_volume": 1200}], vehicle_capacity=1000)


if __name__ == "__main__":
    unittest.main()
