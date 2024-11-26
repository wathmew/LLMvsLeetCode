from collections import deque

class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        def is_valid(s: str) -> bool:
            """
            Checks if a string is valid by counting the number of opening and closing parentheses.
            Returns True if the string is valid, False otherwise.
            """
            count = 0
            for char in s:
                if char == '(':
                    count += 1
                elif char == ')':
                    count -= 1
                if count < 0:
                    return False
            return count == 0

        result = []
        queue = deque([(s, 0)])
        visited = set([s])
        found = False

        while queue:
            current, removals = queue.popleft()
            if is_valid(current):
                result.append(current)
                found = True
            elif found:
                continue

            for i in range(len(current)):
                if current[i] not in '()':
                    next_str = current[:i] + current[i+1:]
                    if next_str not in visited:
                        visited.add(next_str)
                        queue.append((next_str, removals + 1))

        return result
