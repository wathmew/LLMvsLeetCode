import json

lc1 = [
"""
def trap(self, height: List[int]) -> int:
        left = 0
        right = len(height) - 1
        left_max = height[left]
        right_max = height[right]
        water = 0

        while left < right:
            if left_max < right_max:
                left += 1
                left_max = max(left_max, height[left])
                water += left_max - height[left]
            else:
                right -= 1
                right_max = max(right_max, height[right])
                water += right_max - height[right]
        
        return water
"""
]

lc2 = [
"""
def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        max_area = 0

        for i in range(len(heights)):
            while stack[-1] != -1 and heights[i] <= heights[stack[-1]]:
                height = heights[stack.pop()]
                width = i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        
        while stack[-1] != -1:
            height = heights[stack.pop()]
            width = len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        return max_area
"""
]

lc3 = [
"""
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def find_mean(nums):
            mid = int(len(nums)/2)
            if len(nums)%2==0:
                return (nums[mid]+nums[mid-1])/2
            else:
                return nums[mid]/1
            
       
        l1= len(nums1)
        l2= len(nums2)
        final_num = [0]*int(((l1+l2)/2)+1)
        mid = len(final_num)

        if l1==0:
            return find_mean(nums2)
        if l2==0:
            return find_mean(nums1)
        i,j,k=0,0,0
        while k<mid:

            if   i<l1 and (j>=l2 or nums1[i]<=nums2[j]):
                final_num[k]=nums1[i]
                i+=1
                k+=1
            elif j<l2 and (i>=l1 or nums2[j]<=nums1[i]):
                final_num[k]=nums2[j]
                j+=1
                k+=1
        
        if (l1+l2)%2==0:
            return sum(final_num[-2:])/2
        else: return sum(final_num[-1:])/1
"""
]

lc4 = [
"""
def minWindow(self, s: str, t: str) -> str:
        if len(s) < len(t):
            return ""
        
        char_count = defaultdict(int)
        for ch in t:
            char_count[ch] += 1
        
        target_chars_remaining = len(t)
        min_window = (0, float("inf"))
        start_index = 0

        for end_index, ch in enumerate(s):
            if char_count[ch] > 0:
                target_chars_remaining -= 1
            char_count[ch] -= 1

            if target_chars_remaining == 0:
                while True:
                    char_at_start = s[start_index]
                    if char_count[char_at_start] == 0:
                        break
                    char_count[char_at_start] += 1
                    start_index += 1
                
                if end_index - start_index < min_window[1] - min_window[0]:
                    min_window = (start_index, end_index)
                
                char_count[s[start_index]] += 1
                target_chars_remaining += 1
                start_index += 1
        
        return "" if min_window[1] > len(s) else s[min_window[0]:min_window[1]+1]
"""
]

lc5 = [
"""
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = []
        q = deque()

        for idx, num in enumerate(nums):
            while q and q[-1] < num:
                q.pop()
            q.append(num)

            if idx >= k and nums[idx - k] == q[0]:
                q.popleft()
            
            if idx >= k - 1:
                res.append(q[0])
        
        return res
"""
]

lc6 = [
"""
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        li=[]
        for i in range(0,len(lists)):
            temp=lists[i]
            while(temp!=None):
                li.append(temp.val)
                temp=temp.next
        li.sort()
        head=ListNode(None)
        for i in li:
            nn=ListNode(i)
            if head==None:
                head=nn
                continue
            temp=head
            while(temp.next!=None):
                temp=temp.next
            temp.next=nn
        print(li)
        return head.next
"""
]

lc7 = [
"""
def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        
        if not head: return None

        if k == 1: 
            return head

        end_check = head        
        for _ in range(k-1):
            if not end_check.next: 
                return head
            end_check = end_check.next
        
        # if not end_check: return head

        blank = ListNode(0)
        p0, p1, count = blank, head, 0

        while count < k:
            curr = p1
            p2 = curr.next
            curr.next = p0
            p0 = p1
            p1 = p2
            count += 1

        head.next = self.reverseKGroup(p2, k)
        head = p0

        return head
"""
]

lc8 = [
"""
def traverse(self, node, storer):
        if not node:
            return 0
        left = self.traverse(node.left, storer)
        right = self.traverse(node.right, storer)
        if left<0:
            left = 0
        if right < 0:
            right = 0
        storer.append(left + right + node.val)
        return max(left,right) + node.val

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        maxStorer = []
        self.traverse(root, maxStorer)
        return max(maxStorer)
"""
]

lc9 = [
"""
def reversePairs(self, nums: List[int]) -> int:
        N = len(nums)
        tot = 0
        sorted_right_side = [nums[-1]*2]
        for j in range(N-2,-1,-1):

            n = nums[j]
            # count of numbers that are < 2 * n
            tot += bisect.bisect_left(sorted_right_side,n)

            bisect.insort(sorted_right_side, n*2)


        return tot
"""
]

lc10 = [
"""
def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
    
    def exist(word: str) -> bool:
        if len(word) > len(board[0]) * len(board):
            return False
        
        def backtracking(i, j, word_idx: int) -> bool:
            if word_idx == len(word):
                return True
            
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0])
                return False # went out of borders

            elif board[i][j] != word[word_idx]:
                return False # current letter does not match
            
            buff, board[i][j] = board[i][j], '#'
            res = (
                backtracking(i + 1, j, word_idx + 1) or
                backtracking(i - 1, j, word_idx + 1) or
                backtracking(i, j + 1, word_idx + 1) or
                backtracking(i, j - 1, word_idx + 1)
            )

            board[i][j] = buff
            return res
        
        for i in range(len(board)):
            for j in range(len(board[i])):
                if backtracking(i, j, 0):
                    return True
        return False

    res = []

    for word in words:
        if exist(word):
            res.append(word)
    return res
"""
]

lc11 = [
"""
def solveNQueens(self, n: int) -> List[List[str]]:
        def backtrack(r):
            if r == n:
                copy = board[:]
                sol = []
                for c in copy:
                    sol.append("".join(c[:]))
                ans.append(sol)
                return

            for c in range(n):
                if c in placedCol or r + c in placedPos or r - c in placedNeg: continue

                board[r][c] = "Q"
                placedCol.add(c)
                placedPos.add(r + c)
                placedNeg.add(r - c)

                backtrack(r + 1)

                board[r][c] = "."
                placedCol.remove(c)
                placedPos.remove(r + c)
                placedNeg.remove(r - c)

        board = [["."] * n for _ in range(n)]
        
        placedCol = set()
        placedPos = set()
        placedNeg = set()
        ans = []
        backtrack(0)
        return ans
"""
]

lc12 = [
"""
    def find(self, i, job, startTime, n, dp):
        if i >= n:
            return 0
        if dp[i] != -1:
            return dp[i]

        index = startTime.index(job[i][1])

        pick = job[i][2] + self.find(index, job, startTime, n, dp)
        notpick = self.find(i + 1, job, startTime, n, dp)
        dp[i] = max(pick, notpick)
        return dp[i]

    def jobScheduling(self, startTime, endTime, profit):
        n = len(startTime)
        job = [[startTime[i], endTime[i], profit[i]] for i in range(n)]
        dp = [-1] * n

        job.sort(key=lambda x: x[1])
        startTime.sort()

        return self.find(0, job, startTime, n, dp)
"""
]

lc13 = [
"""
 def ladderLength(self, beginWord:str,endWord:str,wordList: List[str])->int:
        q = deque();
        q.append((beginWord,1));
        hmap = defaultdict(int)
        for word in wordList: hmap[word] = 1;
        hmap[beginWord] = 0
        while( len(q) > 0 ):
            #print(q);
            word = q.popleft();
            wordl = list(word[0])
            if( word[1] > 1 and word[0] == endWord):
                return word[1]
            for i in range(0,len(wordl)):
                t = wordl[i]
                for j in range(0,ord('z')-ord('a') + 1):
                    wordl[i] = chr( ord('a') + j );
                    tword = ''.join(wordl)
                    if( hmap[tword] == 1 and tword != word[0] ):
                        q.append( (tword,word[1]+1) );
                        hmap[tword] = 0;
                wordl[i] = t
                    
        return 0;
"""
]

lc14 = [
"""
def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        ans = [-1]*len(queries)

        intervals.sort()
        qidxs = list(range(len(queries)))
        qidxs.sort(key=lambda i: queries[i])

        k = 0 # next interval index to push
        ivals = [] # (size, start, end) for all intervals containing next query, min heap by size
        for i in qidxs:
            q = queries[i]

            while ivals and ivals[0][2] < q: heappop(ivals)
            
            while k < len(intervals) and intervals[k][0] <= q:
                if intervals[k][1] >= q:
                    # skip intervals that start later, but have already ended
                    # (we'd pop them to maintain RI anyway)
                    heappush(ivals, (intervals[k][1]-intervals[k][0]+1, intervals[k][0], intervals[k][1]))
                k += 1

            if ivals:
                ans[i] = ivals[0][0]

        return ans
"""
]

lc15 = [
"""
def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        flight_map = collections.defaultdict(list)
        result = []

        # Populate the flight map with each departure and arrival
        for departure, arrival in tickets:
            flight_map[departure].append(arrival)

        # Sort each list of destinations in reverse lexicographical order
        for departure in flight_map:
            flight_map[departure].sort(reverse=True)

        # Perform DFS traversal
        def dfsTraversal(current):
            destinations = flight_map[current]

            # Traverse all destinations in the order of their lexicographical
            # sorting
            while destinations:
                next_destination = destinations.pop()
                dfsTraversal(next_destination)

            # Append the current airport to the result after all destinations
            # are visited
            result.append(current)

        dfsTraversal("JFK")
        return result[::-1]

"""
]

lc16 = [
"""
def swimInWater(self, grid: List[List[int]]) -> int:
        N = len(grid)
        visit = set()
        directions = [[0,1],[0,-1],[1,0],[-1,0]]
        minH = [[grid[0][0],0,0]]
        visit.add((0,0))
        while minH:
            t,r,c = heapq.heappop(minH)
            if r == N-1 and c == N-1:
                return t
            for dr,dc in directions:
                neiR , neiC = r+dr , c+dc
                if (neiR < 0 or neiC < 0 or neiR >= N or neiC >= N or (neiR,neiC) in visit):
                    continue
                visit.add((neiR,neiC))
                heapq.heappush(minH,[max(t,grid[neiR][neiC]), neiR, neiC])
"""
]

lc17 = [
"""
def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        dp = {}
        rows, cols = len(matrix), len(matrix[0])
        nbr = [[0,1],[0,-1],[1,0],[-1,0]]

        def longest_path(r1, c1):
            if (r1,c1) in dp:
                return dp[(r1,c1)]
            max_path = 1
            for [dr,dc] in nbr:
                nr, nc = r1+dr, c1+dc
                if nr>=0 and nr<rows and nc>=0 and nc<cols:
                    if matrix[nr][nc] > matrix[r1][c1]:
                        max_path = max(max_path, 1+longest_path(nr,nc))
            dp[(r1,c1)] = max_path
            return dp[(r1,c1)]

        ans = 1
        for r in range(rows):
            for c in range(cols):
                path_l = longest_path(r,c)
                ans = max(ans, path_l)
        return ans
"""
]

lc18 = [
"""
def numDistinct(self, s: str, t: str) -> int:
        @cache
        def rec(i,j):
            if i == n:
                return 1
            if j == m:
                return 0
            ans = 0
            if t[i] == s[j]:
                ans += rec(i+1,j+1)
            ans += rec(i,j+1)
            return ans
        n,m = len(t),len(s)
        return rec(0,0)      
"""
]

lc19 = [
"""
def maxCoins(self, nums: List[int]) -> int:

        nums.insert(0,1)
        nums.append(1)

        def recursive(i, j, memo):
            if i == j:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            max_cost = float('-inf')
            for k in range(i, j):
                curr_cost = nums[i-1] * nums[k] * nums[j]
                left_cost = recursive(i, k, memo)
                right_cost = recursive(k+1, j, memo)
                max_cost = max(max_cost, curr_cost + left_cost + right_cost)
            memo[(i,j)] = max_cost
            return max_cost
        
        memo = {}
        return recursive(1, len(nums)-1, memo)
"""
]

lc20 = [
"""
def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for j in range(2, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2] or (dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.'))
                else:
                    dp[i][j] = dp[i - 1][j - 1] and (s[i - 1] == p[j - 1] or p[j - 1] == '.')
        return dp[m][n]
"""
]

lc21 = [
"""
def solveSudoku(self, board: List[List[str]]) -> None:
        
        # Do not return anything, modify board in-place instead.
        

        def isgood(board, row, col, i):
            for j in board[row]:
                if j == str(i):
                    return False

            for j in range(9):
                if board[j][col] == str(i):
                    return False

            temprow = (row // 3) * 3
            tempcol = (col // 3) * 3

            for a in range(temprow, temprow + 3):
                for b in range(tempcol, tempcol + 3):
                    if board[a][b] == str(i):
                        return False

            return True

        dot = 0
        for line in board:
            for i in line:
                if i == ".":
                    dot += 1

        def help(board, row, col):

            if row == 9:
                return True

            nextrow = row
            nextcol = col

            if col == 9:
                return help(board, row + 1, 0)

            if board[row][col] != ".":
                return help(board, row, col + 1)

            for i in range(1, 10):
                if isgood(board, row, col, i):
                    board[row][col] = str(i)
                    if help(board, nextrow, nextcol):
                        return True
                    else:
                        board[row][col] = "."

        help(board, 0, 0)
"""
]

lc22 = [
"""
def minCut(self, s: str) -> int:
        n = len(s)
        
        # Step 1: Preprocess the string to determine which substrings are palindromes
        dp = [[False]*n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        for i in range(n-1):
            if s[i] == s[i+1]:
                dp[i][i+1] = True
        for l in range(3, n+1):
            for i in range(n-l+1):
                j = i+l-1
                if s[i] == s[j] and dp[i+1][j-1]:
                    dp[i][j] = True
        
        # Step 2: Use dynamic programming to determine the minimum cuts needed
        cuts = list(range(n))
        for i in range(1, n):
            if dp[0][i]:
                cuts[i] = 0
            else:
                for j in range(i):
                    if dp[j+1][i]:
                        cuts[i] = min(cuts[i], cuts[j]+1)
        
        # Step 3: Return the final answer
        return cuts[-1]
"""
]

lc23 = [
"""
def maxProfit(self, k: int, prices: List[int]) -> int:
        if k == 0: return 0
        dp = [[1000, 0] for _ in range(k + 1)]
        for price in prices:
            for i in range(1, k + 1):
                dp[i][0] = min(dp[i][0], price - dp[i - 1][1])
                dp[i][1] = max(dp[i][1], price - dp[i][0])
        return dp[k][1]
"""
]

lc24 = [
"""
def calculate(self, s: str) -> int:
        number = 0
        sign_value = 1
        result = 0
        operations_stack = []

        for c in s:
            if c.isdigit():
                number = number * 10 + int(c)
            elif c in "+-":
                result += number * sign_value
                sign_value = -1 if c == '-' else 1
                number = 0
            elif c == '(':
                operations_stack.append(result)
                operations_stack.append(sign_value)
                result = 0
                sign_value = 1
            elif c == ')':
                result += sign_value * number
                result *= operations_stack.pop()
                result += operations_stack.pop()
                number = 0

        return result + number * sign_value
"""
]

lc25 = [
"""
def removeInvalidParentheses(self, s: str) -> List[str]:
        def is_valid(s):
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
                    continue
                next_str = current[:i] + current[i+1:]
                if next_str not in visited:
                    visited.add(next_str)
                    queue.append((next_str, removals + 1))
    
        return result
"""
]

lc26 = [
"""
def maxCoins(self, nums: List[int]) -> int:

        nums.insert(0,1)
        nums.append(1)

        def recursive(i, j, memo):
            if i == j:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            max_cost = float('-inf')
            for k in range(i, j):
                curr_cost = nums[i-1] * nums[k] * nums[j]
                left_cost = recursive(i, k, memo)
                right_cost = recursive(k+1, j, memo)
                max_cost = max(max_cost, curr_cost + left_cost + right_cost)
            memo[(i,j)] = max_cost
            return max_cost
        
        memo = {}
        return recursive(1, len(nums)-1, memo)
"""
]

lc27 = [
"""
def countSmaller(self, nums: List[int]) -> List[int]:
        l = len(nums)
        
        ans = array('i', bytes(len(nums) << 2))
        
        nums2 = array('h')

        for i in range(l-1,-1,-1):
          n = nums.pop()
          lpos = bisect_left(nums2, n)
          ans[i] = lpos
          nums2.insert(lpos, n)

        return ans
"""
]

lc28 = [
"""
def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        def get_max_subseq(nums, k):
            stack = []
            for i, num in enumerate(nums):
                while stack and len(nums) - i + len(stack) > k and stack[-1] < num:
                    stack.pop()
                if len(stack) < k:
                    stack.append(num)
            return stack
        
        def merge(nums1, nums2):
            merged = []
            i, j = 0, 0
            while i < len(nums1) or j < len(nums2):
                if i >= len(nums1):
                    merged.append(nums2[j])
                    j += 1
                elif j >= len(nums2):
                    merged.append(nums1[i])
                    i += 1
                elif nums1[i:] > nums2[j:]:
                    merged.append(nums1[i])
                    i += 1
                else:
                    merged.append(nums2[j])
                    j += 1
            return merged
        
        ans = []
        for i in range(max(0, k - len(nums2)), min(len(nums1), k) + 1):
            j = k - i
            subseq1 = get_max_subseq(nums1, i)
            subseq2 = get_max_subseq(nums2, j)
            merged = merge(subseq1, subseq2)
            ans = max(ans, merged)
        return ans
"""
]

lc29 = [
"""
def countRangeSum(self, nums: List[int], lower: int, upper: int) -> int:
        prefix = nums.copy()
        for i in range(1, len(nums)):
            prefix[i] += prefix[i-1]
        self.count = 0
        for n in prefix:
            if lower <= n <= upper:
                self.count += 1
        
        def merge(left_arr, right_arr):
            start, end = 0, 0 
            for i in range(len(left_arr)):
                while start < len(right_arr) and right_arr[start] - left_arr[i] < lower:
                    start += 1
                while end < len(right_arr) and right_arr[end] - left_arr[i] <= upper:
                    end += 1
                self.count += end - start

            l, r = 0, 0
            sorted_arr = []
            while l < len(left_arr) and r < len(right_arr):
                if left_arr[l] < right_arr[r]:
                    sorted_arr.append(left_arr[l])
                    l += 1
                else:
                    sorted_arr.append(right_arr[r])
                    r += 1
            return sorted_arr + left_arr[l:] + right_arr[r:]
        
        def divide(arr):
            if len(arr) <= 1: return arr
            mid = len(arr)//2
            left_arr = divide(arr[:mid])
            right_arr = divide(arr[mid:])
            return merge(left_arr, right_arr)

        divide(prefix)
        return self.count
        
"""
]

lc30 = [
"""
def minPatches(self, nums: List[int], n: int) -> int:
        miss = 1
        result = 0
        i = 0

        while miss <= n:
            if i < len(nums) and nums[i] <= miss:
                miss += nums[i]
                i += 1
            else:
                miss += miss
                result += 1

        return result
"""
]

lc31 = [
"""
def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        
        res = []		
		# Perform LIS
        for _, h in envelopes:
            l,r=0,len(res)-1
			# find the insertion point in the Sort order
            while l <= r:
                mid=(l+r)>>1
                if res[mid]>=h:
                    r=mid-1
                else:
                    l=mid+1        
            idx = l
            if idx == len(res):
                res.append(h)
            else:
                res[idx]=h
        return len(res)
"""
]

lc32 = [
"""
def canCross(self, stones: List[int]) -> bool:

        if stones[1] != 1:
            return False

        def dfs(i, k):
            if i == len(stones) - 1:
                return True

            if (i, k) in dp:
                 return dp[(i, k)]

            res = False
            for j in range(i + 1, len(stones)):
                if stones[i] + k == stones[j]:
                    res = res or dfs(j, k)
                if stones[i] + k + 1 == stones[j]:
                    res = res or dfs(j, k + 1)
                if stones[i] + k - 1 == stones[j]:
                    res = res or dfs(j, k - 1)

            dp[(i, k)] = res
            return res
        
        dp = {}
        return dfs(1, 1)
"""
]

lc33 = [
"""
def strongPasswordChecker(self, password: str) -> int:
        unique_chars = set(password)
        password_length = len(password)

        required_categories = 3 - (bool(unique_chars & set(ascii_lowercase)) + 
                                   bool(unique_chars & set(ascii_uppercase)) +
                                   bool(unique_chars & set('0123456789')))

        if password_length < 6:
            return max(6 - password_length, required_categories)

        consecutive_repeats = [len(list(g)) for _, g in groupby(password)]
        long_repeats = [length for length in consecutive_repeats if length > 2]

        if password_length > 20:
            long_repeats = [(length % 3, length) for length in long_repeats]
            heapify(long_repeats)
            for _ in range(password_length - 20):
                if not long_repeats:
                    break
                _, length = heappop(long_repeats)
                if length > 3:
                    heappush(long_repeats, ((length - 1) % 3, length - 1))
            long_repeats = [length for _, length in long_repeats]

        return max(required_categories, sum(length // 3 for length in long_repeats)) + max(0, password_length - 20)
"""
]

lc34 = [
"""
 def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        max_profit=[]
        min_capital=[(c,p) for c,p in zip(capital,profits)]
        heapq.heapify(min_capital) 
        for i in range(k):
            while min_capital and min_capital[0][0]<=w:
                c,p=heapq.heappop(min_capital) 
                heapq.heappush(max_profit,-1*p)
            if not max_profit:
                break
            w+=-1*heapq.heappop(max_profit)
        return w
"""
]

lc35 = [
"""
def findRotateSteps(self, ring: str, key: str) -> int:
        n = len(ring)
        matches = {}
        for i in range(n):
            matches.setdefault(ring[i], []).append(i)
        
        pos_cost = [(0, 0)]
        for ch in key:
            pos_cost_curr = []
            for curr_pos in matches[ch]:
                curr_cost = float('inf')
                for pos, cost in pos_cost:
                    clkwise_trans_cost = abs(pos - curr_pos)
                    temp_cost = cost + min(clkwise_trans_cost, n - clkwise_trans_cost)
                    curr_cost = min(curr_cost, temp_cost)
                pos_cost_curr.append((curr_pos, curr_cost))
            pos_cost = pos_cost_curr
        
        min_cost = float('inf')
        for pos, cost in pos_cost:
            min_cost = min(min_cost, cost)
        
        return min_cost + len(key)
"""
]

lc36 = [
"""
def findMinMoves(self, machines: List[int]) -> int:
        n = len(machines)
        total_dresses = sum(machines)
        
        if total_dresses % n != 0:
            return -1
        
        target_dresses = total_dresses // n
        moves = 0
        dresses_so_far = 0
        
        for i in range(n):
            dresses_so_far += machines[i] - target_dresses
            moves = max(moves, abs(dresses_so_far), machines[i] - target_dresses)
        
        return moves
"""
]

lc37 = [
"""
def findIntegers(self, n: int) -> int:
        f = [1,2]
        for i in range(2,30):
            f.append(f[-1] + f[-2])
        
        ans = last_seen = 0

        for i in range(29,-1,-1):
            if (1 << i) & n:
                ans += f[i]
                if last_seen :
                    return ans
                last_seen = 1
            else:
                last_seen = 0
        return ans + 1
"""
]

lc38 = [
"""
def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key=lambda c: c[1])
        A, curr = [], 0
        for dur, ld in courses:
            heapq.heappush(A,-dur)
            curr += dur
            if curr > ld: curr += heapq.heappop(A)
        return len(A)
"""
]

lc39 = [
"""
    def strangePrinter(self, s: str) -> int:
        n = len(s)
        dp = [[-1] * n for _ in range(n)]
        return self.Util(0, n - 1, s, dp)

    def Util(self, i: int, j: int, s: str, dp: list) -> int:
        if i > j:
            return 0

        if dp[i][j] != -1:
            return dp[i][j]

        first_letter = s[i]
        # If the current character is not repeated in the rest of the string
        answer = 1 + self.Util(i + 1, j, s, dp)
        for k in range(i + 1, j + 1):
            # If repeated then update the answer
            if s[k] == first_letter:
                # Splitting from i -> k - 1 (remove the last character)
                # and from k + 1 -> j             
                better_answer = self.Util(i, k - 1, s, dp) + self.Util(k + 1, j, s, dp)
                answer = min(answer, better_answer)
        dp[i][j] = answer
        return answer
"""
]

lc40 = [
"""
def evaluate(self, expression: str) -> int:

        def parse(e: str) -> list[str]:
            tokens, s, parenthesis = [], '', 0

            for c in e:
                if c == '(':
                    parenthesis += 1
                elif c == ')':
                    parenthesis -= 1
                    
                if parenthesis == 0 and c == ' ':
                    tokens.append(s)
                    s = ''
                else:
                    s += c

            if s: 
                tokens.append(s)
            
            return tokens

        def evaluate_expression(e: str, prevScope: dict) -> int:
            if e[0].isdigit() or e[0] == '-':
                return int(e)
            if e in prevScope:
                return prevScope[e]

            scope = prevScope.copy()
            nextExpression = e[e.index(' ') + 1:-1]
            tokens = parse(nextExpression)

            if e[1] == 'a':
                return evaluate_expression(tokens[0], scope) + evaluate_expression(tokens[1], scope)
            if e[1] == 'm':
                return evaluate_expression(tokens[0], scope) * evaluate_expression(tokens[1], scope)

            for i in range(0, len(tokens) - 2, 2):
                scope[tokens[i]] = evaluate_expression(tokens[i + 1], scope)

            return evaluate_expression(tokens[-1], scope)

        return evaluate_expression(expression, {})
"""
]

lc41 = [
"""
    def crackSafe(self, n: int, k: int) -> str:
        if n == 1:
            return ''.join(map(str, range(k)))
        
        seen = set()
        result = []
        
        start_node = "0" * (n - 1)
        self.dfs(start_node, k, seen, result)
        
        return "".join(result) + start_node

    def dfs(self, node, k, seen, result):
        for i in range(k):
            edge = node + str(i)

            if edge not in seen:
                seen.add(edge)

                self.dfs(edge[1:], k, seen, result)
                result.append(str(i))
"""
]

lc42 = [
"""
def slidingPuzzle(self, board: List[List[int]]) -> int:

        state = (*board[0], *board[1])
        queue, seen, cnt = deque([state]), set(), 0

        while queue:

            for _ in range(len(queue)):
                state = list(queue.popleft())
                idx = state.index(0)
                if state == [1,2,3,4,5,0]: return cnt

                for i in moves[idx]:
                    curr = state[:]
                    curr[idx], curr[i] = curr[i], 0
                    curr = tuple(curr)
                    if curr in seen: continue
                    queue.append(curr)
                    seen.add(curr)

            cnt+= 1

        return -1
"""
]

lc43 = [
"""
def hitBricks(self, grid: List[List[int]], hits: List[List[int]]) -> List[int]:
        rows, cols = len(grid), len(grid[0])
        def neighbours(r, c):
            for new_r, new_c in [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]:
                if not (0 <= new_r < rows and 0 <= new_c < cols): continue
                yield new_r, new_c

        def make_stable(to_do):
            changed = 0
            for r, c in to_do:
                grid[r][c] = 2
            while to_do:
                r, c = to_do.pop()
                for new_r, new_c in neighbours(r, c):
                    if grid[new_r][new_c] == 1:
                        changed += 1
                        grid[new_r][new_c] = 2
                        to_do.append((new_r, new_c))
            return changed
        
        def can_be_stable(r, c):
            if r == 0: return True
            for new_r, new_c in neighbours(r, c):
                if grid[new_r][new_c] == 2:
                    return True
            return False


        for r, c in hits:
            grid[r][c] = -grid[r][c]
        
        stable = [(0, c) for c in range(cols) if grid[0][c] == 1]
        make_stable(stable)
        result = [0 for _ in range(len(hits))]
        for p, (r, c) in enumerate(reversed(hits), start = 1):
            if grid[r][c] == 0: continue
            if not can_be_stable(r, c):
                grid[r][c] = 1 
                continue 
            changed = make_stable([(r, c)])
            result[-p] = changed

        return result
"""
]

lc44 = [
"""
def racecar(self, target: int) -> int:
        queue = deque([(0,0,1)])
        visited = set()
        while queue:
            moves, position,speed = queue.popleft()
            if position == target:
                return moves
            if (position,speed) in visited:
                continue
            else:
                visited.add((position,speed))
                queue.append((moves+1,position+speed,speed*2))
                if (position+speed > target and speed > 0) or (position+speed < target and speed < 0):
                    speed = -1 if speed > 0 else 1
                    queue.append((moves+1,position,speed))
        return 0
"""
]

lc45 = [
"""
def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
        # 9:37
        total_wage, total_quality, max_heap = float('inf'), 0, []
        for ratio, q in sorted([(w/q, q) for w, q in zip(wage, quality)]):
            total_quality += q
            heappush(max_heap, -q)
            if len(max_heap) == k:
                total_wage = min(total_wage, total_quality * ratio)
                total_quality += heappop(max_heap)
        return total_wage
"""
]

lc46 = [
"""
def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        def GCD(a: int, b: int) -> int:
            x, y = a, b
            while y:
                x, y = y, x % y
            return x
                
        def LCM(a: int, b: int) -> int:
            return a * b / GCD(a, b)

        lcm = LCM(a, b)
        low, high = 2, 10 ** 14
        while low < high:
            mid = (low + high) // 2
            if mid // a + mid // b - mid // lcm < n:
                low = mid + 1
            else:
                high = mid
        
        return low % (10 ** 9 + 7)
"""
]

lc47 = [
"""
def orderlyQueue(self, s: str, k: int) -> str:
        if k > 1: return ''.join(sorted(s))
        m = s
        for i in range(1, len(s)):
            # print(s[i: ]+s[: i])
            m = min(m, s[i: ]+s[: i])
        return m
"""
]

lc48 = [
"""
def movesToStamp(self, stamp: str, target: str) -> List[int]:
        m, n = len(stamp), len(target)
        stamp = list(stamp)
        target = list(target)
        
        # Helper function to check if we can stamp at position i
        def can_stamp(i):
            for j in range(m):
                if target[i + j] != '?' and target[i + j] != stamp[j]:
                    return False
            return True
        
        # Helper function to apply stamp at position i
        def apply_stamp(i):
            for j in range(m):
                target[i + j] = '?'
        
        # Track visited positions and the stamping process
        stamped = [False] * n
        result = []
        queue = deque()
        
        # Initial queue fill
        for i in range(n - m + 1):
            if can_stamp(i):
                queue.append(i)
                apply_stamp(i)
                result.append(i)
                stamped[i] = True
        
        # Process the queue
        while queue:
            pos = queue.popleft()
            for i in range(max(0, pos - m + 1), min(n - m + 1, pos + m)):
                if can_stamp(i):
                    if not stamped[i]:
                        queue.append(i)
                        apply_stamp(i)
                        result.append(i)
                        stamped[i] = True
        
        # Verify that all characters in target are stamped
        if all(c == '?' for c in target):
            return result[::-1]
        else:
            return []
"""
]

lc49 = [
"""
def oddEvenJumps(self, arr: List[int]) -> int:
        n = len(arr)
        
        O, E = 0, 1

        dp = [[False for _ in range(n)], [False for _ in range(n)]]
        dp[O][-1], dp[E][-1] = True, True

        index_map = {arr[-1]: n-1}
        sorted_list = [arr[-1]]
        for i, x in list(enumerate(arr[:-1]))[::-1]:
            index = bisect.bisect_left(sorted_list, x)

            if index != len(sorted_list) and sorted_list[index] == x:
                dp[O][i] = dp[E][index_map[sorted_list[index]]]
                dp[E][i] = dp[O][index_map[sorted_list[index]]]
            else:
                if index != len(sorted_list):
                    dp[O][i] = dp[E][index_map[sorted_list[index]]]
                if index != 0:
                    dp[E][i] = dp[O][index_map[sorted_list[index-1]]]
            sorted_list.insert(index, x)
            index_map[x] = i
        return sum(dp[O])
"""
]

lc50 = [
"""
def maxSizeSlices(self, slices: List[int]) -> int:
        n = len(slices)  
        m = n // 3  

        def dp(slices_subset):
            k = len(slices_subset)
            dp_table = [[0] * (m + 1) for _ in range(k + 1)]
            for i in range(1, k + 1):
                for j in range(1, min(i, m) + 1):
                    dp_table[i][j] = max(dp_table[i - 1][j], dp_table[i - 2][j - 1] + slices_subset[i - 1])
            return dp_table[k][m]

        return max(dp(slices[:-1]), dp(slices[1:]))
"""
]

lc_refs = [globals()[f"lc{i}"] for i in range(1, 51)]

with open("lc_hard.json", "r") as file:
    data = json.load(file)

def convert_to_json_string(multiline_string):
    return multiline_string.replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"')

lc_refs_single = [[convert_to_json_string(ref_string) for ref_string in lc_ref]  for lc_ref in lc_refs]

for i, entry in enumerate(data):
    if "ref" in entry:
        entry["ref"] = []
        refs = lc_refs_single[i]
        for ref in refs:
            entry["ref"].append(ref)
    else:
        print(f"json ref formatted incorrectly on entry {i}")
        break

with open("lc_hard.json", "w") as file:
    json.dump(data, file, indent=4)

print("updated lc hard dataset references")