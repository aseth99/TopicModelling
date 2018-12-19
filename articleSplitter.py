import json
import csv
 
# txtFileName = "newsDatatest.txt"

articles = []
titles = []
counter = 0

jsonFileToBeOpened = "news.json"
for line in open(jsonFileToBeOpened, 'r'):
	articles.append(json.loads(line))

for line in articles:
	counterStr = str(counter)+".txt"

	titleData = line.get('title')
	textData = line.get('text')
	str2 = textData.replace("\n", " ")
	with open(counterStr, 'a') as the_file:
		the_file.write(titleData + "\n")
		the_file.write(str2 + "\n")
	counter = counter + 1
	
# with open('newsDatatest.txt') as infile, open('sanitizedOutput.txt', 'w') as outfile:
#     for line in infile:
#         if not line.strip(): continue  # skip the empty line
#         outfile.write(line)  # non-empty line. Write it to output
