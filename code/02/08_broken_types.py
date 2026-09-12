# expect-fail
# Failure 2 of 4: the right value in the wrong type. This one comes from CSV files
# constantly, because everything in a CSV is text until you convert it.

qty = "480"          # read from a file, so it is text
unit_price = 12.50

print("About to multiply", qty, "by", unit_price)
print(qty * unit_price)
