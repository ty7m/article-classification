# Overview

Implementation of article classification using naive bayes and display on web app.

## Setup

```bash
docker build -t project .
docker run -it -p 80:8000 project /bin/bash
```

## Usage

### Basic

- Collect data and pickle them

```bash
python manage.py collect_data
```

- Launch server

```bash
python manage.py runserver 0.0.0.0:8000
```

- Access to http://127.0.0.1/app/input and input the URL you want to know the category

### Check accuracy

I used accuracy as an evaluation index this time. The accuracy is defined by the following equation.
<p align='center'>
  <img src="https://latex.codecogs.com/gif.latex?Accuracy&space;=&space;\frac{TP&plus;TN}{TP&plus;FP&plus;FN&plus;TN}&space;\\[&plus;0.5em]&space;TP&space;:&space;\textrm{True&space;Positive}&space;\\&space;TN&space;:&space;\textrm{True&space;Negative}&space;\\&space;FP&space;:&space;\textrm{False&space;Positive}&space;\\&space;FN&space;:&space;\textrm{False&space;Negative}" />
</p>

You can check naive bayes accuracy as below.
```bash
python manage.py naive_bayes
```

You can also check accuracy of another method to improve accuracy. The method is explained below.

```bash
python manage.py improved_method
```

## Improved method

Naive bayes assumes that each word is independent, so the accuracy is not so high. Consider the following functions based on Bayesian theorem for each category and consider that x belongs to the category where the function is the largest. Assume that x is based on a normal distribution.

<p align='center'>
  <img src="https://latex.codecogs.com/gif.latex?g_i(x)&space;=&space;\textrm{log}\&space;p(x|w_i)&space;&plus;&space;\textrm{log}P(w_i)&space;\\[&plus;0.5em]&space;p(x)&space;=&space;\frac{1}&space;{(2\pi)^{d/2}|\sum|^{1/2}}\textrm{exp}\left(-\frac{1}{2}(x-\mu)^T{\sum}^{-1}(x-\mu)\right)&space;\\[&plus;0.5em]&space;\therefore&space;g_i(x)&space;=&space;-\frac{1}{2}(x-\mu_i)^T{\sum}_i^{-1}(x-\mu_i)&space;-&space;\frac{d}{2}\textrm{log}2\pi&space;-&space;\frac{1}{2}\textrm{log}|{\sum}_i|&space;&plus;&space;\textrm{log}P(w_i)" />
</p>
When the constant term is delated, this function is expressed as follow.(Some terms are omitted because the function diverges.) 
<p align='center'>
  <img src="https://latex.codecogs.com/gif.latex?g_i(\bm{x})&space;=&space;\bm{x}^T\bm{W}_i\bm{x}&plus;\bm{w}_i^T\bm{x}&plus;\bm{w}_{i0}" />
</p>
<p align='center'>
  <img src="https://latex.codecogs.com/gif.latex?W_i&space;=&space;-\frac{1}{2}{{\sum}_{i}}^{-1}" />
</p>
<p align='center'>
  <img src="https://latex.codecogs.com/gif.latex?\bm{w}_i&space;=&space;{\bm{{\sum}_{I}}}^{-1}\bm{\mu}_{i}" />
</p>
<p align='center'>
  <img src="https://latex.codecogs.com/gif.latex?\bm{w}_{i0}&space;&=&space;{{\bm{\mu}}_i}^T\bm{W}_i\bm{\mu}_{i}&space;&plus;\textrm{log}{P(\omega_i)}" />
</p>

## Code check
 
You can check if the code comply with pep8. If "passing" is displayed, the code meet the conventions.

[![Build Status](https://travis-ci.org/ty7m/article-classification.svg?branch=master)](https://travis-ci.org/ty7m/article-classification)  

