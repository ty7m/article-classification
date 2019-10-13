#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:18:12 2019

@author: takatoyamada
"""
from django.urls import path

from . import views

urlpatterns = [
    path('input', views.input),
    path('output', views.output),
]
