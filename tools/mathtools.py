from langchain.tools import Tool
import math

# === Math functions ===

def add(input: str) -> str:
    a, b = map(float, input.split(","))
    return str(a + b)

def subtract(input: str) -> str:
    a, b = map(float, input.split(","))
    return str(a - b)

def multiply(input: str) -> str:
    a, b = map(float, input.split(","))
    return str(a * b)

def divide(input: str) -> str:
    a, b = map(float, input.split(","))
    if b == 0:
        return "Error: Division by zero"
    return str(a / b)

def power(input: str) -> str:
    a, b = map(float, input.split(","))
    return str(a ** b)

def sqrt(input: str) -> str:
    a = float(input.strip())
    if a < 0:
        return "Error: Negative number"
    return str(math.sqrt(a))

def percentage(input: str) -> str:
    part, total = map(float, input.split(","))
    if total == 0:
        return "Error: Total is zero"
    return str((part / total) * 100) + "%"

def factorial(input: str) -> str:
    n = int(input.strip())
    if n < 0:
        return "Error: Negative number"
    return str(math.factorial(n))

def max_value(input: str) -> str:
    values = list(map(float, input.split(",")))
    return str(max(values))

def min_value(input: str) -> str:
    values = list(map(float, input.split(",")))
    return str(min(values))

def mean_value(input: str) -> str:
    values = list(map(float, input.split(",")))
    return str(sum(values) / len(values))

# === Tools list ===

math_tools = [
    Tool(name="add_numbers", func=add, description="Add two numbers. Format: 'a,b'"),
    Tool(name="subtract_numbers", func=subtract, description="Subtract two numbers (a - b). Format: 'a,b'"),
    Tool(name="multiply_numbers", func=multiply, description="Multiply two numbers. Format: 'a,b'"),
    Tool(name="divide_numbers", func=divide, description="Divide two numbers (a ÷ b). Format: 'a,b'"),
    Tool(name="power_numbers", func=power, description="Raise a number to a power (a^b). Format: 'a,b'"),
    Tool(name="sqrt_number", func=sqrt, description="Calculate the square root of a number. Format: 'a'"),
    Tool(name="percentage_of_total", func=percentage, description="Calculate what percentage a is of b. Format: 'a,b'"),
    Tool(name="factorial_number", func=factorial, description="Calculate the factorial of a number (n!). Format: 'n'"),
    Tool(name="maximum_value", func=max_value, description="Get the highest value in a list of numbers. Format: 'a,b,c,...'"),
    Tool(name="minimum_value", func=min_value, description="Get the smallest value in a list of numbers. Format: 'a,b,c,...'"),
    Tool(name="average_value", func=mean_value, description="Calculate the average (mean) of a list of numbers. Format: 'a,b,c,...'"),
]
