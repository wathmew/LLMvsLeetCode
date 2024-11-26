# test_code.py
from generated_code import Solution

def test_remove_invalid_parentheses():
    solution = Solution()
    
    # Test case 1
    print(solution.removeInvalidParentheses("()())()"))
    assert solution.removeInvalidParentheses("()())()") == ["(())()","()()()"]

    # Test case 2
    assert solution.removeInvalidParentheses("(a)())()") == ["(a())()","(a)()()"]

    # Test case 3
    assert solution.removeInvalidParentheses(")(") == [""]
