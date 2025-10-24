# print ("hello world")

def second_largest(numbers):
    if len(numbers) < 2:
        return None  # Not enough elements
    
    first = second = float('-inf')
    
    for num in numbers:
        if num > first:
            second = first
            first = num
        elif first > num > second:
            second = num
    
    return second if second != float('-inf') else None


# Example usage
nums = [10, 20, 4, 45, 99]
result = second_largest(nums)

if result is not None:
    print("The second largest number is:", result)
else:
    print("Not enough distinct elements.")
