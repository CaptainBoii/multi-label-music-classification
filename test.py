import os

files = map(lambda file: file[:-4], os.listdir("./Spectrograms/Processed/AudioFiles/"))
spectro = map(lambda file: file[:-4], os.listdir("./Spectrograms/Spectrograms/"))

result = list(set(files) - set(spectro))
result.sort()
print(result)
