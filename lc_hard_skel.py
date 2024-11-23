import json

lc1 = """
def trap(self, height: List[int]) -> int:
"""

lc2 = """
def largestRectangleArea(self, heights: List[int]) -> int:
"""

lc3 = """
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
"""

lc4 = """
def minWindow(self, s: str, t: str) -> str:
"""

lc5 = """
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
"""

lc6 = """
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
"""

lc7 = """
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
"""

lc8 = """
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def maxPathSum(self, root: Optional[TreeNode]) -> int:
"""

lc9 = """
def reversePairs(self, nums: List[int]) -> int:
"""

lc10 = """
def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
"""

lc11 = """
def solveNQueens(self, n: int) -> List[List[str]]:
"""

lc12 = """
def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
"""

lc13 = """
def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
"""

lc14 = """
def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
"""

lc15 = """
def findItinerary(self, tickets: List[List[str]]) -> List[str]:
"""

lc16 = """
def swimInWater(self, grid: List[List[int]]) -> int:
"""

lc17 = """
def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
"""

lc18 = """
def numDistinct(self, s: str, t: str) -> int:
"""

lc19 = """
def maxCoins(self, nums: List[int]) -> int:
"""

lc20 = """
def isMatch(self, s: str, p: str) -> bool:
"""

lc21 = """
def solveSudoku(self, board: List[List[str]]) -> None:
    #Do not return anything, modify board in-place instead.
"""

lc22 = """
def minCut(self, s: str) -> int:
"""

lc23 = """
def maxProfit(self, k: int, prices: List[int]) -> int:
"""

lc24 = """
def calculate(self, s: str) -> int:
"""

lc25 = """
def removeInvalidParentheses(self, s: str) -> List[str]:
"""

lc26 = """
def maxCoins(self, nums: List[int]) -> int:
"""

lc27 = """
def countSmaller(self, nums: List[int]) -> List
"""

lc28 = """
def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
"""

lc29 = """
def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
"""

lc30 = """
def minPatches(self, nums: List[int], n: int) -> int:
"""

lc31 = """
def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
"""

lc32 = """
def canCross(self, stones: List[int]) -> bool:
"""

lc33 = """
def strongPasswordChecker(self, password: str) -> int:
"""

lc34 = """
def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
"""

lc35 = """
def findRotateSteps(self, ring: str, key: str) -> int:
"""

lc36 = """
def findMinMoves(self, machines: List[int]) -> int:
"""

lc37 = """
def findIntegers(self, n: int) -> int:
"""

lc38 = """
def scheduleCourse(self, courses: List[List[int]]) -> int:
"""

lc39 = """
def strangePrinter(self, s: str) -> int:
"""

lc40 = """
def evaluate(self, expression: str) -> int:
"""

lc41 = """
def crackSafe(self, n: int, k: int) -> str:
"""

lc42 = """
def slidingPuzzle(self, board: List[List[int]]) -> int:
"""

lc43 = """
def hitBricks(self, grid: List[List[int]], hits: List[List[int]]) -> List[int]:
"""

lc44 = """
def racecar(self, target: int) -> int:
"""

lc45 = """
def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
"""

lc46 = """
def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
"""

lc47 = """
def orderlyQueue(self, s: str, k: int) -> str:
"""

lc48 = """
def movesToStamp(self, stamp: str, target: str) -> List[int]:
"""

lc49 = """
def oddEvenJumps(self, arr: List[int]) -> int:
"""

lc50 = """
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