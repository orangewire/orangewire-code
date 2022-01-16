#!/usr/bin/python3

import random
import sys
import string


def luhn_sum(digits):
    """
    Helper function to compute the sum in the Luhn algorithm
    """

    s = 0
    for idx, digit in enumerate(map(int, digits)):
        mu = 1 if idx % 2 else 2
        n = mu * digit
        s += n - 9 if n > 9 else n

    return s


def generate_card(issuer):
    """
    Generate a valid credit card number by specifying an issuer
    """

    prefixes = {
        "americanexpress": [34, 37],
        "mastercard":      list(range(51, 56)),
        "visa":            [4]
    }

    lengths = {
        "americanexpress": [15],
        "mastercard":      [16],
        "visa":            [13, 16]
    }

    if not issuer in prefixes.keys():
        raise Exception("Unknown issuer")

    # Get a prefix and a card number length at random
    prefix = str(random.choice(prefixes[issuer]))
    length = random.choice(lengths[issuer])
    # Generate a random body
    body_length = length - len(prefix) - 1
    body = ''.join(random.choice(string.digits)
                   for _ in range(body_length))
    # Concatenate prefix and body
    payload = prefix + body

    # Calculate the check digit using Luhn's algorithm
    s = luhn_sum(payload[::-1])
    c = (10 - s % 10) % 10

    # Concatenate check digit
    return payload + str(c)


def luhn_check(card_number):
    """
    Validate a credit card number
    """

    c = int(card_number[-1])
    s = luhn_sum(card_number[len(card_number)-2::-1])
    return (s + c) % 10 == 0


def main(argv):
    # Generate a few card numbers and perform a Luhn check on each of them
    issuers = ["americanexpress", "mastercard", "visa"]
    labels = random.choices(issuers, k=20)
    nums = [generate_card(issuer) for issuer in labels]

    for label, num in zip(labels, nums):
        print(
            f'{label.ljust(15)}: {num} -> {"Valid" if luhn_check(num) else "Invalid"}')


if __name__ == '__main__':
    main(sys.argv[1:])
