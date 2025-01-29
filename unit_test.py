import unittest
from unittest.mock import patch
from knuckleGame import play_knucklebones, roll_dice, place_choice, clear_column, calculate_column_score, display_board

class TestKnuckleGame(unittest.TestCase):

    def test_calculate_column_score(self):
        # Test with all zeros
        column = [0, 0, 0]
        result = calculate_column_score(column)
        self.assertEqual(result, 0)

        # Test with no duplicates
        column = [1, 2, 3]
        result = calculate_column_score(column)
        self.assertEqual(result, 1 + 2 + 3)

        # Test with one duplicate
        column = [1, 1, 2]
        result = calculate_column_score(column)
        self.assertEqual(result, 1*2*2 + 2)

        # Test with all duplicates
        column = [2, 2, 2]
        result = calculate_column_score(column)
        self.assertEqual(result, 2*3*3)

        # Test with mixed duplicates
        column = [3, 3, 1]
        result = calculate_column_score(column)
        self.assertEqual(result, 3*2*2 + 1)

        # Test with another set of mixed duplicates
        column = [4, 4, 4]
        result = calculate_column_score(column)
        self.assertEqual(result, 4*3*3)

        # Test with different values
        column = [5, 5, 6]
        result = calculate_column_score(column)
        self.assertEqual(result, 5*2*2 + 6)

        # Test with another set of different values
        column = [6, 6, 3]
        result = calculate_column_score(column)
        self.assertEqual(result, 6*2*2 + 3)


        # Test with different values
        column = [5, 5, 6]
        result = calculate_column_score(column)
        self.assertEqual(result, 5*2*2 + 6)

        # Test with another set of different values
        column = [6, 6, 1]
        result = calculate_column_score(column)
        self.assertEqual(result, 6*2*2 + 1)

        # Test with all same values
        column = [3, 3, 3]
        result = calculate_column_score(column)
        self.assertEqual(result, 3*3*3)

        # Test with mixed values
        column = [1, 2, 3]
        result = calculate_column_score(column)
        self.assertEqual(result, 1 + 2 + 3)

        # Test with two same values and one different
        column = [4, 4, 2]
        result = calculate_column_score(column)
        self.assertEqual(result, 4*2*2 + 2)

        # Test with one value and two zeros
        column = [6, 0, 0]
        result = calculate_column_score(column)
        self.assertEqual(result, 6)

        # Test with two different values and one zero
        column = [2, 5, 0]
        result = calculate_column_score(column)
        self.assertEqual(result, 2 + 5)

        # Test with three different values
        column = [1, 4, 6]
        result = calculate_column_score(column)
        self.assertEqual(result, 1 + 4 + 6)

        # Test with maximum values
        column = [6, 6, 6]
        result = calculate_column_score(column)
        self.assertEqual(result, 6*3*3)

        # Test with all same values
        column = [3, 3, 3]
        result = calculate_column_score(column)
        self.assertEqual(result, 3*3*3)

        # Test with mixed values
        column = [1, 2, 3]
        result = calculate_column_score(column)
        self.assertEqual(result, 1 + 2 + 3)

        # Test with two same values and one different
        column = [4, 4, 2]
        result = calculate_column_score(column)
        self.assertEqual(result, 4*2*2 + 2)

        # Test with one value and two zeros
        column = [6, 0, 0]
        result = calculate_column_score(column)
        self.assertEqual(result, 6)

        # Test with two different values and one zero
        column = [2, 5, 0]
        result = calculate_column_score(column)
        self.assertEqual(result, 2 + 5)

        # Test with three different values
        column = [1, 4, 6]
        result = calculate_column_score(column)
        self.assertEqual(result, 1 + 4 + 6)

        # Test with maximum values
        column = [6, 6, 6]
        result = calculate_column_score(column)
        self.assertEqual(result, 6*3*3)

        # Test with minimum values
        column = [1, 1, 1]
        result = calculate_column_score(column)
        self.assertEqual(result, 1*3*3)

        # Test with alternating values
        column = [2, 3, 2]
        result = calculate_column_score(column)
        self.assertEqual(result, 2*2*2 + 3)

        # Test with descending values
        column = [6, 5, 4]
        result = calculate_column_score(column)
        self.assertEqual(result, 6 + 5 + 4)

        # Test with ascending values
        column = [1, 2, 3]
        result = calculate_column_score(column)
        self.assertEqual(result, 1 + 2 + 3)

if __name__ == '__main__':
    unittest.main()