# Names point at values. Two names can point at the same value, and then changing it
# through one name changes what the other one sees.

history = [{"role": "system", "content": "Answer in one sentence."}]

# This does not copy the list. It gives the same list a second name.
conversation = history
conversation.append({"role": "user", "content": "Summarise 2024 Q3."})
print("len(history):", len(history), "   same object?", conversation is history)

# list(...) makes a new list — a shallow copy.
history = [{"role": "system", "content": "Answer in one sentence."}]
copied = list(history)
copied.append({"role": "user", "content": "Summarise 2024 Q3."})
print("after copy + append -> history:", len(history), " copied:", len(copied))

# Shallow means the new list holds the SAME dictionaries.
copied[0]["content"] = "Answer at length."
print("history[0] now says:", repr(history[0]["content"]))

# deepcopy copies all the way down.
import copy                                                     # noqa: E402

history = [{"role": "system", "content": "Answer in one sentence."}]
deep = copy.deepcopy(history)
deep[0]["content"] = "Answer at length."
print("after deepcopy, history[0]:", repr(history[0]["content"]))


# The most famous version of this bug: a mutable default argument. The default list is
# created ONCE, when the function is defined, and shared by every call that omits it.
def add_message(text, messages=[]):                               # noqa: B006
    messages.append(text)
    return messages


print("\nfirst call: ", add_message("hello"))
print("second call:", add_message("goodbye"))   # nobody passed the first message in


def add_message_fixed(text, messages=None):
    messages = [] if messages is None else messages
    messages.append(text)
    return messages


print("fixed, first: ", add_message_fixed("hello"))
print("fixed, second:", add_message_fixed("goodbye"))

# Numbers, strings and tuples cannot be changed in place, so they cannot surprise you.
label = "Q3"
other = label
other += " 2024"            # makes a NEW string and points `other` at it
print("\nlabel:", repr(label), "  other:", repr(other))
