import json
import csv
import os.path

save_path = '/Users/aman/Desktop/Hammer/TopicModelling/Articles'

 # txtFileName = "newsDatatest.txt"

articles = []
titles = []
counter = 1

jsonFileToBeOpened = "news.json"
for line in open(jsonFileToBeOpened, 'r'):
	articles.append(json.loads(line))

for line in articles:
	counterStr = str(counter)+".txt"
	completeName = os.path.join(save_path,counterStr)
	urlData = line.get('url')
	titleData = line.get('title')
	textData = line.get('text')
	str2 = textData.replace("\n", " ")
	with open(completeName, 'w') as the_file:
		the_file.write("ARTICLE ID: " + str(counter) + "\n\n")
		the_file.write("TITLE: \n" + titleData + "\n\n")
		the_file.write("ARTICLE: \n" + str2 + "\n\n")
		the_file.write("URL: \n" + urlData + "\n\n")
		the_file.close()
	counter = counter + 1
