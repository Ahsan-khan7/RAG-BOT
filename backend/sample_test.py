def is_prime(number):
    if number <= 1:
        return False
    elif number == 2:
        return True
    elif number % 2 == 0:
        return False

    # Only check up to square root of number
    for i in range(3, int(number**0.5) + 1, 2):
        if number % i == 0:
            return False
    return True

# Take input from user
try:
    #num = int(input("Enter a number to check if it's prime: "))
    num=0
    
    if is_prime(num):
        print(f"{num} is a prime number.")
    else:
        print(f"{num} is not a prime number.")
except ValueError:
    print("Please enter a valid integer.")
