import json
t=0
import csv 
from langdetect import detect
filename = "Mined_data.csv" 
import re
lis=[]
# used for preprocessing posts
fields = ['text']
def preprocess(text):
	try:
		if detect(text)=='en':
			text=text.split(' ')
			# hashtags=[]
			processed_text=''
			for word in text:
				if '#' in word:
					# hashtags.append(word[1:])
					# processed_text+=word[1:]+' '
					pass
				elif word.isalpha():
					processed_text+=word+' '
			if processed_text=='':
				return None
			return processed_text
	except:
		return text

with open('scraper_posts.json') as json_file:
	data = json.load(json_file)
	l=1
	for s in data['text']:
		print(s)
		processed_tex=preprocess(s)
		if(processed_tex!=None and len(processed_tex)>4):
			lis.append([processed_tex])
	
	# for d in data:
	# 	try:
	# 		books = d['text']
	# 		print(books)
			# books.replace("\n", "")
			# pattern=r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
			# match=re.search(pattern,books)
			# if(match!=None):
			# 	continue
			
		# 	processed_tex=preprocess(books)
		# 		# print(processed_tex)
		# 	if(processed_tex!=None and len(processed_tex)>4):
		# 		lis.append([processed_tex])
		# except:
		# 	pass
		# print(books)
# print(len(lis))
# print(lis)
# # print(lis[0])
with open(filename, 'w') as csvfile:
	# creating a csv writer object  
	csvwriter = csv.writer(csvfile)  
	csvwriter.writerow(fields)
	# writing the data rows  
	csvwriter.writerows(lis) 
