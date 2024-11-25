import json

lc1 = """
class Solution:
    def trap(self, height: List[int]) -> int:
"""

lc2 = """
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
"""

lc3 = """
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
"""

lc4 = """
class Solution:
    def minWindow(self, s: str, t: str) -> str:
"""

lc5 = """
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
"""

lc6 = """
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
"""

lc7 = """
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
"""

lc8 = """
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
"""

lc9 = """
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
"""

lc10 = """
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
"""

lc11 = """
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
"""

lc12 = """
class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
"""

lc13 = """
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
"""

lc14 = """
class Solution:
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
"""

lc15 = """
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
"""

lc16 = """
class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
"""

lc17 = """
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
"""

lc18 = """
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
"""

lc19 = """
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
"""

lc20 = """
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
"""

lc21 = """
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        #Do not return anything, modify board in-place instead.
"""

lc22 = """
class Solution:
    def minCut(self, s: str) -> int:
"""

lc23 = """
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
"""

lc24 = """
class Solution:
    def calculate(self, s: str) -> int:
"""

lc25 = """
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
"""

lc26 = """
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
"""

lc27 = """
class Solution:
    def countSmaller(self, nums: List[int]) -> List
"""

lc28 = """
class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
"""

lc29 = """
class Solution:
    def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
"""

lc30 = """
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
"""

lc31 = """
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
"""

lc32 = """
class Solution:
    def canCross(self, stones: List[int]) -> bool:
"""

lc33 = """
class Solution:
    def strongPasswordChecker(self, password: str) -> int:
"""

lc34 = """
class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
"""

lc35 = """
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
"""

lc36 = """
class Solution:
    def findMinMoves(self, machines: List[int]) -> int:
"""

lc37 = """
class Solution:
    def findIntegers(self, n: int) -> int:
"""

lc38 = """
class Solution:
    def scheduleCourse(self, courses: List[List[int]]) -> int:
"""

lc39 = """
class Solution:
    def strangePrinter(self, s: str) -> int:
"""

lc40 = """
class Solution:
    def evaluate(self, expression: str) -> int:
"""

lc41 = """
class Solution:
    def crackSafe(self, n: int, k: int) -> str:
"""

lc42 = """
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
"""

lc43 = """
class Solution:
    def hitBricks(self, grid: List[List[int]], hits: List[List[int]]) -> List[int]:
"""

lc44 = """
class Solution:
    def racecar(self, target: int) -> int:
"""

lc45 = """
class Solution:
    def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
"""

lc46 = """
class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
"""

lc47 = """
class Solution:
    def orderlyQueue(self, s: str, k: int) -> str:
"""

lc48 = """
class Solution:
    def movesToStamp(self, stamp: str, target: str) -> List[int]:
"""

lc49 = """
class Solution:
    def oddEvenJumps(self, arr: List[int]) -> int:
"""

lc50 = """
class Solution:
    def maxSizeSlices(self, slices: List[int]) -> int:
"""

lc_skel = [globals()[f"lc{i}"] for i in range(1, 51)]

with open("lc_hard.json", "r") as file:
    data = json.load(file)

def convert_to_json_string(multiline_string):
    return multiline_string.replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"')

lc_skel_single = [convert_to_json_string(lc_skel_multiline) for lc_skel_multiline in lc_skel]

for i, entry in enumerate(data):
    if "skeleton" in entry:
        entry["skeleton"] = lc_skel_single[i]
    else:
        print(f"json skeleton formatted incorrectly on entry {i}")
        break

with open("lc_hard.json", "w") as file:
    json.dump(data, file, indent=4)

print("updated lc hard dataset skeletons")