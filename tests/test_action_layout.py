"""Configuration boundaries and video/local index semantics for compact action."""

import unittest

from fasterwam.models.wan22.action_layout import resolve_action_layout


class ActionLayoutTests(unittest.TestCase):
    def test_default_is_original_thirty_layer_identity(self):
        conditions = [0, 4, 8, 12, 16, 20, 24, 28]
        layout = resolve_action_layout(30, conditions)
        self.assertEqual(layout.action_layer_indices, tuple(range(30)))
        self.assertEqual(layout.action_condition_layers, tuple(conditions))
        self.assertEqual(layout.noncondition_layers_per_interval, (3,) * 7)
        self.assertEqual(layout.trailing_noncondition_layers, 1)
        self.assertEqual(layout.leading_noncondition_layers, 0)
        self.assertEqual(layout.num_action_layers, 30)

    def test_uniform_and_per_interval_counts(self):
        conditions = [0, 4, 8, 12, 16, 20, 24, 28]
        layout = resolve_action_layout(30, conditions, 1)
        self.assertEqual(layout.num_action_layers, 16)
        self.assertEqual(layout.action_condition_layers, tuple(range(0, 16, 2)))
        self.assertEqual(layout.action_layer_indices, (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29))
        self.assertEqual(resolve_action_layout(30, conditions, 2).num_action_layers, 23)
        reduced = resolve_action_layout(30, [0, 12, 24, 28], [2, 3, 1], 0)
        self.assertEqual(reduced.action_layer_indices, (0, 1, 2, 12, 13, 14, 15, 24, 25, 28))
        self.assertEqual(reduced.action_condition_layers, (0, 3, 7, 9))
        self.assertEqual(resolve_action_layout(30, [0, 12, 24, 28], 1).num_action_layers, 8)

    def test_leading_layers_and_tail_limits(self):
        layout = resolve_action_layout(9, [2, 5], 1, 2)
        self.assertEqual(layout.action_layer_indices, (0, 1, 2, 3, 5, 6, 7))
        self.assertEqual(layout.action_condition_layers, (2, 4))
        self.assertEqual(layout.leading_noncondition_layers, 2)
        self.assertEqual(resolve_action_layout(5, [0, 4], 1, 0).action_layer_indices, (0, 1, 4))
        self.assertEqual(resolve_action_layout(1, [0], [], 0).num_action_layers, 1)

    def test_leading_count_defaults_and_local_condition_remapping(self):
        for count, prefix in ((None, (0, 1, 2, 3, 4)), (5, (0, 1, 2, 3, 4)),
                              (2, (0, 1)), (0, ())):
            with self.subTest(count=count):
                layout = resolve_action_layout(12, [5, 9], 1, 1, count)
                self.assertEqual(layout.action_layer_indices, prefix + (5, 6, 9, 10))
                self.assertEqual(layout.action_condition_layers, (len(prefix), len(prefix) + 2))
                self.assertEqual(layout.leading_noncondition_layers, len(prefix))
        self.assertEqual(resolve_action_layout(6, [5], [], 0, 0).action_layer_indices, (5,))
        self.assertEqual(resolve_action_layout(1, [0], [], 0, 0).action_condition_layers, (0,))

    def test_invalid_leading_counts_fail(self):
        for first, value in ((0, 1), (5, 6), (5, -1), (5, True), (5, False),
                             (5, 1.0), (5, "2"), (5, [2])):
            with self.subTest(first=first, value=value), self.assertRaisesRegex(
                ValueError, "action_leading_noncondition_layers",
            ):
                resolve_action_layout(12, [first, 9], leading_noncondition_layers=value)

    def test_invalid_conditions_and_counts_fail_early(self):
        cases = [
            (30, [], None, None), (30, [0, 4, 4], None, None),
            (30, [4, 0], None, None), (30, [-1, 4], None, None),
            (30, [0, 30], None, None), (30, [0, 1], None, None),
            (30, [0, True], None, None), (30, [0, 4.0], None, None),
            (30, "0,4", None, None), (0, [0], None, None),
            (True, [0], None, None), (30.0, [0], None, None),
            (30, [0, 4], 0, None), (30, [0, 4], -1, None),
            (30, [0, 4], True, None), (30, [0, 4], 1.5, None),
            (30, [0, 4], "1", None), (30, [0, 4], 4, None),
            (30, [0, 4], [], None), (30, [0, 4], [1, 1], None),
            (30, [0, 4], [0], None), (30, [0, 4], [False], None),
            (30, [0, 12, 24, 28], [1, 1, 4], None),
            (30, [0, 28], 1, -1), (30, [0, 28], 1, 2),
            (30, [0, 28], 1, True), (30, [0, 28], 1, 0.5),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_action_layout(*args)


if __name__ == "__main__":
    unittest.main()
