class_names = ['Hello', 'I Love You', 'Okay', 'Please', 'Thank you', 'Yes']  # or use train_ds.class_names

with open(r"C:\Harsh Works\code\American Sign Language\labels.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")
