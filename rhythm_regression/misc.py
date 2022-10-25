def rational_numbers(n):
    """
    Returns the first n rational numbers.
    """

    if n == 0:
        return []

    nums = [1] 
    nums_set = set([1])
    denominator = 2
    while len(nums) < n:
        for numerator in range(1, denominator):
            if (numerator / denominator) not in nums_set:
                nums.append(numerator / denominator)
                nums_set.add(numerator / denominator)

            if len(nums) >= n:
                return nums

            if (denominator / numerator) not in nums_set:
                nums.append(denominator / numerator)
                nums_set.add(denominator / numerator)

            if len(nums) >= n:
                return nums

        denominator += 1

    return nums