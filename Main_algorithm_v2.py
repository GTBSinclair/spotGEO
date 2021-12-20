#!/usr/bin/env python
# coding: utf-8

# In[1]:


#####
# Script: Giovanni Tognini Bonelli Sinclair
# Date: 07/12/21
#####

class Brightness_analysis:
    
    def normalize(kernel):
        for i in range(len(kernel)): 
            for j in range(len(kernel[i])):
                kernel[i][j] = kernel[i][j]*2
        return kernel

    
    def average_calc(kernel,area,i,j): # NEED TO RE-WRITE THIS AS A GET_AVG_AREA_# FUNCTION and then apply the rule below for any anomalously bright or dark pixels
        if area == 1:
            avg_ring = round((kernel[i][j]+sum(kernel[i-1][j-1:j+2])+sum(kernel[i+1][j-1:j+2])+kernel[i][j+1]+kernel[i][j-1])/9,5)
        if area == 2:
            avg_ring = round((sum(kernel[i-2][j-2:j+3])+sum(kernel[i+2][j-2:j+3])+kernel[i-1][j-2]+kernel[i][j-2]+kernel[i+1][j-2]+kernel[i-1][j+2]+kernel[i][j+2]+kernel[i+1][j+2]+sum(kernel[i-3][j-3:j+4])+sum(kernel[i+3][j-3:j+4])+kernel[i-2][j-3]+kernel[i-1][j-3]+kernel[i][j-3]+kernel[i+1][j-3]+kernel[i+2][j-3]+kernel[i-2][j+3]+kernel[i-1][j+3]+kernel[i][j+3]+kernel[i+1][j+3]+kernel[i+2][j+3])/40,5)
        return avg_ring


    def append_averages(kernel): # 'kernel' is the input image scanned by the algorithm
        
        averages = []
#         kernel = Brightness_analysis.normalize(kernel)
        
        for i in range(len(kernel)): 
            for j in range(len(kernel[i])):
                try:
                    avg_area_1 = Brightness_analysis.average_calc(kernel,1,i,j)
                    avg_area_2 = Brightness_analysis.average_calc(kernel,2,i,j)
        #             print('worked')
                    averages.append([i, j, avg_area_1, avg_area_2])
                except IndexError as error:
                    continue
        return averages
    
    
    def analyse_dataset(features,labels,ID,ID2):
        
        list_of_GEOsat_averages = []

        if ID == 0:
            print('Analysing training dataset...')
            
            if ID2 == 4:
                    img = kernel2
                    averages = Brightness_analysis.append_averages(img)
                    list_of_GEOsat_averages.append(averages)
            
            
            for i in range(len(labels)):
                if ID2 == 0:
                    if labels[i] == 0:
                        img = features[i].reshape((7,7))
                        averages = Brightness_analysis.append_averages(img)
                        list_of_GEOsat_averages.append([averages,labels[i],i])
                        plt.imshow(img, cmap = 'gray', vmin=0, vmax=1)
                        
                if ID2 == 1:
                    if labels[i] == 1:
                        img = features[i].reshape((7,7))
                        averages = Brightness_analysis.append_averages(img)
                        list_of_GEOsat_averages.append([averages,labels[i],i])
                        # plt.imshow(img, cmap = 'gray', vmin=0, vmax=1)
                        
                if ID2 == 3:
                    img = features[i].reshape((7,7))
                    averages = Brightness_analysis.append_averages(img)
                    list_of_GEOsat_averages.append([averages,labels[i],i])
                    
                    
        elif ID == 1:
            print('Analysing test dataset...')
            
            # NOT COMPLETED here
            
            averages = Brightness_analysis.append_averages(img)
            list_of_GEOsat_averages.append(averages)
                    
        return list_of_GEOsat_averages
        
        
    def rearrange_results(features,labels,ID,ID2):
        
        results = Brightness_analysis.analyse_dataset(features,labels,ID,ID2)
        
        new_list_avgs = []

        for t in range(len(results)):
            for i in range(len(results[0][0])):
                if results[t][0][i][0] == 3 and results[t][0][i][1] == 3:
                    new_list_avgs.append([results[t][0][i],results[t][1],results[t][2]])
                    
        return new_list_avgs
                    
        
    def generate_list_of_areas(features,labels,ID,ID2):
        
        bb = Brightness_analysis.rearrange_results(features,labels,ID,ID2)
        
        decrease_in_brightness_avg = []
        list_of_areas_1 = []
        list_of_areas_2 = []

        for i in range(len(bb)):
            if bb[i][0][2]>bb[i][0][3] or bb[i][0][2]<=bb[i][0][3]: 
                decrease_in_brightness_avg.append(round(bb[i][0][2]-bb[i][0][3],5))
                list_of_areas_1.append(round(bb[i][0][2],5))
                list_of_areas_2.append(round(bb[i][0][3],5))
                    
        return [list_of_areas_1,list_of_areas_2]

    
    def calculate_delta_areas(features,labels,ID,ID2):
        
        master_list_of_areas = Brightness_analysis.generate_list_of_areas(features,labels,ID,ID2)
        
        Delta_area = []

        for i in range(len(master_list_of_areas[0])):
            Delta_area.append(round(master_list_of_areas[0][i]-master_list_of_areas[1][i],5)) # avg area 1 - avg area 2 for false satellites

        return [Delta_area]


    def classify(features,labels,ID,ID2,threshold):

        master_delta_areas = Brightness_analysis.calculate_delta_areas(features,labels,ID,ID2)
        
        assigned_labels = []

        for i in range(len(master_delta_areas[0])):
            if master_delta_areas[0][i] <= threshold:
                assigned_labels.append(0)
            elif master_delta_areas[0][i] > threshold and master_delta_areas[0][i] < 0.5:
                assigned_labels.append(1)
                
        return [assigned_labels]

