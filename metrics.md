1. PRs merge rate per user: how many prs get merged / how many prs they make
    - Gives an assessment of the quality of their PRs
    - high -> good quality and low friction code
    - low -> not great quality and high friction code, PRs often rejected
2. Total count of PRs merged by user: how many prs they made that were closed
    - Signal on how much someone ships code
    - high -> productive
    - low -> less productive
<!-- 3. Total Impact Ratio (how many prs merged / total num of prs (irrespective of branch)) -->
3. Total number of PRs merged into main by user (Product Impact Ratio): how many prs into main / total num of prs by user
    - Signal on how much of their code gets pushed into deployment
    - high -> high contribution
    - low -> low contribution
4. Time to Merge : closed_at - created_at, then take median of these values
    - Signal on how quickly their PRs are reviewed and merged
    - high -> low friction review and deployment
    - low -> more time to review, etc
5. Days_active: how many of the 90 days were they actively making PRs
    - Signal on how active the developers are
    - high num of prs -> very active

6. PRs in recent days (last 30 days)

7. Stability: variance in their time_to_merge
    - low variance -> consistently giving same time to deliver
    - high variance -> unstable in the time of merging

8. Activity: total_prs * (1/median_time_to_merge)
    - high-> highly efficient, either large num of prs or v small time_to_merge
