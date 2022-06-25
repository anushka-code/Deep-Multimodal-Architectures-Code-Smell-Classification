def TokenizeGroups(text): #tokenizes as per CamelCase RegEx and converts to lowercase
  tokenizer = RegexpTokenizer('[a-zA-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))') # RegEx pattern for CamelCase 
  useful_text = tokenizer.tokenize(text) 
  useful_text = [x.lower() for x in useful_text]
  return useful_text
  
  def LoopOver(list): #loops over all the samples to tokenize all the strings in each word group
  for i in range(0,1146):
    list[i] = TokenizeGroups(list[i][0])
    i=i+1
  return list

class_words_sem = LoopOver(class_words)
method_words_sem = LoopOver(method_words)

def WordList(list1,list2): #concatenate the words of each project, package, class and method
  final_list = []
  for i in range(0,1146):
    x = list1[i] + list2[i]
    final_list.append(x)
  res = [' '.join(ele) for ele in final_list]
  return res

word_groups = WordList(class_words_sem, method_words_sem)
