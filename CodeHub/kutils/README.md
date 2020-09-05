# utils

## data
   * onehotencode(xs)
   * transpose(mat)
   * scan(not yet developed) : gives distribution of numbers

## files
   * get_dirs(dir_s, recursive=None) : support recursive search
   * get_files(dir_s, recursive=None)
   * list_dirs(dir_s, recursive=None) : return dictionary form
   * dump files : in format of (yaml, json, pk, npz)
   * load files 

## lists
   * count(arr1, ret = None) : return a dictionary of element counting 
   * split(arr1, step): split a list with given step
   * isiterable(arr1): check arr1 is iterable
   * flatten(arr1, depth = 1, level = 0, ret_list = None)
   * 추가 예정 :  
       * sorteddictionary(modify)  
       * list --> indexed dictionary  

## string
   * filter a string array with given predicate
   * 추가 예정
      * strip and chomp

## web (추가 예정)
   * crawl

## test (추가 예정)
  * stopwatch
  

## nlp (추가 예정)
   * tfidf
   * text processing

## plot (추가 예정)
   * explorer data 

## report (추가 예정)
   * data_report(data)
        ** make a reports with 
        ** histogram of variables
        ** correlation plots
        ** variable stats

   * learning_report(plot learning process)

## preprocess (추가 예정)
   * missing value
   * normalize

## db (추가 예정)
   * sql 접속 간단히

## print (추가 예정)
       

## time series (추가 예정)
  * smoothing
  * interpolation
  * windows
  * DTW
  * transformation
  * deviations

