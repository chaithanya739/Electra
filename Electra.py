
import scipy as sp
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Pool
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
import os

#parameter initialization 
global Number_of_ANDgates_per_PE, total_PEs, cycle_ratio_CEtoLE, Multoadd_ratio, Number_of_mul_counters, Number_of_adders, Mul, Add, status, completion_status
Number_of_ANDgates_per_PE = 2 
#provide even number
total_PEs = 20000 #total number of PEs to be utilized
cycle_ratio_CEtoLE = 4 #ratio of cycles of computation and logical operations
Multoadd_ratio = 2 #ratio of multiplier to adder elements
Number_of_mul_counters =  cycle_ratio_CEtoLE #The number of multiplier counters should be same as that of ratio of cycles of computation and logical operations
Number_of_adders = Multoadd_ratio #Number of adders should be same as that of the ratio of multiplier to adder element
cycle_count = 0 #Intializing the cycle count. The variable is used to determine the number of cycles required to execute the Electra operation

class Multiplier: #class of muliplier elements
    def __init__(self):
        pass
    def operation(self,non_zero_A,non_zero_B):
        operation = non_zero_A * non_zero_B
        return operation

class Adder: # class of adder elements
    def __init__(self):
        pass
    def operation(self,mul_A_output,mul_B_output):
        operation = mul_A_output + mul_B_output
        return operation

class logical_element: #class of logical elements
    def __init__(self):
        pass
    def operation(self,bit_value_A,bit_value_B): 
        operation = bit_value_A and bit_value_B
        return operation

def Scheduler(C0,C1,non_zero_buffer_A,non_zero_buffer_B,i,j,Number_of_ANDgates_per_PE,len_bitmap_buffer_A,len_bitmap_buffer_B): # low level scheduler
    global Mul_cnt, Mul, result_mul, add_cnt, add_mult_cnt, add, result_add, multiplier_result, adder_result, bitmap_result, non_zero_index_B_cnt,non_zero_index_A_cnt
    global total_PEs, cycle_ratio_CEtoLE, Multoadd_ratio, Number_of_mul_counters, Number_of_adders,non_zero_A_temp_check, multiplier_adder_result,result_mul_add,status,cycle_count,PE_utilization

    bitmap_result[i][j] = C0 or C1 # check for the result
    for s in range(cycle_ratio_CEtoLE):
        if (Mul_cnt[s] != 0):
            PE_utilization = PE_utilization + 1


    cycle_count = cycle_count + 1 # cycle count parameter checking

    for s in range(cycle_ratio_CEtoLE):
            if (Mul_cnt[s] != 0):
                Mul_cnt[s] = Mul_cnt[s] + 1
            if(Mul_cnt[s] >= cycle_ratio_CEtoLE):
                Mul_cnt[s] = 0
                if(result_mul_add[s] == 0):
                    multiplier_result.append(result_mul[s])
                    result_mul[s] = 0
                else:
                    multiplier_adder_result.append(result_mul_add[s])
                    
                    result_mul_add[s] = 0
                    if (len(multiplier_adder_result) == Number_of_ANDgates_per_PE):
                        multiplier_result.append(0)

    for g in range(Multoadd_ratio):
        if (add_cnt[g] != 0):
            add_cnt[g] = add_cnt[g] + 1
        if (add_cnt[g] == Multoadd_ratio):
            adder_result.append(result_add[g])
            result_add[g] = 0
            add_cnt[g] = 0

    for g in range(Multoadd_ratio):
        if (add_cnt[g] == 0 and len(multiplier_adder_result) == 2):
                
                result_add[g] = add[g].operation(multiplier_adder_result[-1],multiplier_adder_result[-2])
                multiplier_adder_result.pop()
                multiplier_adder_result.pop()
                add_cnt[g] = add_cnt[g] + 1
                
                break
                    
            


    if(C0 == 1 and C1 == 0):
        non_zero_A_temp_check[0] = non_zero_A_temp_check[0] + 1         

        # parallelize the Multipliers

        #indexing of non_zero buffer according to the bit map operation
           
        for s in range(cycle_ratio_CEtoLE):
            if Mul_cnt[s] == 0:
                
                result_mul[s] = Mul[s].operation(non_zero_buffer_A[0][non_zero_index_A_cnt[0]],non_zero_buffer_B[0][non_zero_index_B_cnt[0]])
                
                non_zero_index_B_cnt[0] = non_zero_index_B_cnt[0] + 1
                Mul_cnt[s] = Mul_cnt[s] + 1 # updating the multiplier counter elements
                break

                
    if(C1 == 1 and C0 == 0):
        non_zero_A_temp_check[1] = non_zero_A_temp_check[1] + 1

        #indexing of non_zero buffer according to the bit map operation


        for s in range(cycle_ratio_CEtoLE):
            if Mul_cnt[s] == 0:            
                result_mul[s] = Mul[s].operation(non_zero_buffer_A[1][non_zero_index_A_cnt[1]],non_zero_buffer_B[1][non_zero_index_B_cnt[1]])
                
                non_zero_index_B_cnt[1] = non_zero_index_B_cnt[1] + 1
                
                Mul_cnt[s] = Mul_cnt[s] + 1
                
                break

    
    if (C1 == 1 and C0 == 1):
        non_zero_A_temp_check[0] = non_zero_A_temp_check[0] + 1
        non_zero_A_temp_check[1] = non_zero_A_temp_check[1] + 1
        if (np.all(Mul_cnt) == True):
            cycle_count = cycle_count + 1

        if (np.all(Mul_cnt) == True):
            for s in range(cycle_ratio_CEtoLE):
                if (Mul_cnt[s] != 0):
                    PE_utilization = PE_utilization + 1
        


        #indexing of non_zero buffer according to the bit map operation
            

        for s in range(cycle_ratio_CEtoLE):
            if Mul_cnt[s] == 0:
                
                result_mul_add[s] = Mul[s].operation(non_zero_buffer_A[0][non_zero_index_A_cnt[0]],non_zero_buffer_B[0][non_zero_index_B_cnt[0]])
                non_zero_index_B_cnt[0] = non_zero_index_B_cnt[0] + 1
                Mul_cnt[s] = Mul_cnt[s] + 1
                
                break

        for s in range(cycle_ratio_CEtoLE):
            if (Mul_cnt[s] != 0):
                Mul_cnt[s] = Mul_cnt[s] + 1
            if(Mul_cnt[s] >= cycle_ratio_CEtoLE):
                Mul_cnt[s] = 0
                if(result_mul_add[s] == 0):
                    multiplier_result.append(result_mul[s])
                    result_mul[s] = 0
                else:
                    multiplier_adder_result.append(result_mul_add[s])
                    
                    result_mul_add[s] = 0
                    if (len(multiplier_adder_result) == Number_of_ANDgates_per_PE):
                        multiplier_result.append(0)

        for s in range(cycle_ratio_CEtoLE):
            if Mul_cnt[s] == 0:
                result_mul_add[s] = Mul[s].operation(non_zero_buffer_A[1][non_zero_index_A_cnt[1]],non_zero_buffer_B[1][non_zero_index_B_cnt[1]])
                non_zero_index_B_cnt[1] = non_zero_index_B_cnt[1] + 1
                Mul_cnt[s] = Mul_cnt[s] + 1
                
                break

        for g in range(Multoadd_ratio):
            if (add_mult_cnt[g] == 0):
                add_mult_cnt[g] = add_mult_cnt[g] + 2 # updating the Mul_add counters
                break


        #indexing of non_zero buffer according to the bit map operation

    if (j == len_bitmap_buffer_B-1):

            for s in range(Number_of_ANDgates_per_PE):
                non_zero_index_B_cnt[s] = 0
                if (non_zero_A_temp_check[s] != 0) :
                    non_zero_index_A_cnt[s] = non_zero_index_A_cnt[s] + 1
                    non_zero_A_temp_check[s] = 0
        


    for g in range(Multoadd_ratio):
        if (add_mult_cnt[g] != 0):
            add_mult_cnt[g] = add_mult_cnt[g] + 1

    for g in range(Multoadd_ratio):
        if (add_mult_cnt[g] == cycle_ratio_CEtoLE+2):
            add_mult_cnt[g] = 0

    

    if (i== len_bitmap_buffer_A-1 and j == len_bitmap_buffer_B-1): #assigning the adder result to the appropriate indicies of the multiplier result
        for s in range(cycle_ratio_CEtoLE):
            if(result_mul[s]!=0):
                multiplier_result.append(result_mul[s])
        
        for g in range(Multoadd_ratio):
            if(result_add[g]!=0):
                adder_result.append(result_add[g])
        for g in range(Multoadd_ratio):
            if(len(multiplier_adder_result)== 2):
                result_add[g] = add[g].operation(multiplier_adder_result[-1],multiplier_adder_result[-2])
                multiplier_adder_result.pop()
                multiplier_adder_result.pop()
                adder_result.append(result_add[g])
                break

        for k in range(len(multiplier_result)):
            if (multiplier_result[k] == 0):
                multiplier_result[k] = adder_result[0]
                adder_result.pop(0)

        for s in range(cycle_ratio_CEtoLE):
            if(Mul_cnt[s]!=0):
                cycle_count = cycle_count + (cycle_ratio_CEtoLE-Mul_cnt[s])
        for k in range(Multoadd_ratio):
            if(add_cnt[k]!=0):
                cycle_count = cycle_count + (Multoadd_ratio-add_cnt[k])
            if(add_mult_cnt[k]!=0):
                cycle_count = cycle_count +2


        
                
    return cycle_count, PE_utilization
    

    # you can print these values if you want to, if you check the functionality of what scheduler actually doing
    """
    print("values for each function calling :")
    print("non_zero_index_A_cnt",non_zero_index_A_cnt)
    print("non_zero_index_B_cnt",non_zero_index_B_cnt)
    print("Mul_cnt",Mul_cnt)
    print("add_cnt",add_cnt)
    print("add_mult_cnt",add_mult_cnt)
    print("bitmap_result",bitmap_result)
    print("adder_result",adder_result)
    print("multiplier_result",multiplier_result)
    print("result_add",result_add)
    print("result_mul",result_mul)
    print("i,j",i,j)
    print("length of bit map buffer", len_bitmap_buffer_A,len_bitmap_buffer_B)
    print("result_mul_add",result_mul_add)
    print("result_add",result_add)
    print("multiplier_adder_result",multiplier_adder_result)
    print("multiplier_adder_result",multiplier_adder_result)
    print("cycle_count",cycle_count)
    print("pid",os.getpid())
    print("            ")
                    
                    
    if (i == len_bitmap_buffer_A-1):
        print("completed") 
    else:
        print("notcompleted")

    """




#processing element funtional block

def Processing_element(Bitmap_A,Bitmap_B,non_zero_A,non_zero_B):
    bitmap_buffer_A = Bitmap_A
    bitmap_buffer_B = Bitmap_B
    non_zero_buffer_A = non_zero_A
    non_zero_buffer_B = non_zero_B
    len_bitmap_buffer_A = len(bitmap_buffer_A)
    len_bitmap_buffer_B = len(bitmap_buffer_B)
    global Mul_cnt, Mul, result_mul, add_cnt, add_mult_cnt, add, result_add, multiplier_result, adder_result, bitmap_result, non_zero_index_B_cnt,non_zero_index_A_cnt,non_zero_A_temp_check
    global Number_of_ANDgates_per_PE, result_mul_add,multiplier_adder_result,cycle_count,PE_utilization
    PE_utilization = 0
    cycle_count = 0

    Mul_cnt = [0 for i in range(cycle_ratio_CEtoLE)]
    result_mul = [0 for i in range(cycle_ratio_CEtoLE)]
    add_cnt = [0 for i in range(Multoadd_ratio)]
    add_mult_cnt = [0 for i in range(Multoadd_ratio)]
    result_add = [0 for i in range(Multoadd_ratio)]
    Mul = [0 for i in range(cycle_ratio_CEtoLE)]
    add = [0 for i in range(Multoadd_ratio)]
    result_mul_add = [0 for i in range(cycle_ratio_CEtoLE)]
    multiplier_adder_result = []

    multiplier_result = []
    adder_result = []
    bitmap_result = [[0]*len(Bitmap_B) for i in range(len(Bitmap_A))]

    non_zero_index_B_cnt = [0 for i in range(Number_of_ANDgates_per_PE)]
    non_zero_index_A_cnt = [0 for i in range(Number_of_ANDgates_per_PE)]
    non_zero_A_temp_check = [0 for i in range(Number_of_ANDgates_per_PE)]

    for k in range(cycle_ratio_CEtoLE):
        Mul[k] = Multiplier()


    for k in range(Multoadd_ratio):
        add[k] = Adder()

    LE1 = logical_element()
    LE2 = logical_element()
    #pipeline this
    for i in range(len(bitmap_buffer_A)):
        #parallel hardware units
        for j in range(len(bitmap_buffer_B)):
            #this loop has to be parallelized in HLS and here we are using multi-threading
            #for k in range(Number_of_ANDgates_per_PE/):
            #the number of C values is determined by number of defined AND gates
            
            C0 = LE1.operation(bitmap_buffer_A[i][0],bitmap_buffer_B[j][0]) #logical element operation
            C1 = LE2.operation(bitmap_buffer_A[i][1],bitmap_buffer_B[j][1])

            #here MUL_1, MUL_2 are the array containing two elements for the inputs to the multiplier units
            

            cycle_count,PE_utilization = Scheduler(C0,C1,non_zero_buffer_A,non_zero_buffer_B,i,j,Number_of_ANDgates_per_PE,len_bitmap_buffer_A,len_bitmap_buffer_B)
            
            # bit_value_result_A is one then k = 0
    Mul_utilization = PE_utilization/(cycle_count*cycle_ratio_CEtoLE)
    return cycle_count,Mul_utilization
            
            




            

def PE_scheduler(Bitmap_A,Bitmap_B,non_zero_A,non_zero_B,Number_of_ANDgates_per_PE):
    global status, completion_status
    Number_of_ANDgates_per_PE
    bitmap_buffer_A = Bitmap_A
    bitmap_buffer_B = Bitmap_B
    print("bitmap_buffer_A",len(bitmap_buffer_A))
    print("bitmap_buffer_B",len(bitmap_buffer_B))
    non_zero_buffer_A = non_zero_A
    non_zero_buffer_B = non_zero_B
    req_PEs = int(len(bitmap_buffer_A[0])/(Number_of_ANDgates_per_PE))
    print("req_PE",req_PEs)
    non_zero_count_A = 0
    non_zero_count_B = 0
    non_zero_A_divisons_per_col = [] 
    non_zero_B_divisons_per_col = []
    buffer_A_per_PE = [[[0 for k in range(Number_of_ANDgates_per_PE)] for j in range(len(bitmap_buffer_A))] for i in range(req_PEs)] #dividing the matrix vertically according to the number of and gates per PE
    buffer_B_per_PE = [[[0 for k in range(Number_of_ANDgates_per_PE)] for j in range(len(bitmap_buffer_B))] for i in range(req_PEs)] # This is for matrix B
    non_zero_A_per_PE = []
    non_zero_B_per_PE = []
    completion_status = []
    for i in range(req_PEs):
        
    #bitvalues and non-zero values assigning for matrix A
         
        for j in range(Number_of_ANDgates_per_PE):
            non_zero_temp_A = []
            non_zero_temp_B = []
            for k in range(len(bitmap_buffer_A)):
                buffer_A_per_PE[i][k][j] = bitmap_buffer_A[k][j+i*Number_of_ANDgates_per_PE]  # allocating the bitmap values to the allocated buffer per PE
                if(bitmap_buffer_A[k][j+i*Number_of_ANDgates_per_PE] != 0):
                    non_zero_temp_A.append(non_zero_buffer_A[non_zero_count_A])
                    non_zero_count_A = non_zero_count_A + 1
            non_zero_A_divisons_per_col.append(non_zero_temp_A)
                


            #bitvalues and non-zero values assigning for matrix B

            for f in range(len(bitmap_buffer_B)):
                buffer_B_per_PE[i][f][j] = bitmap_buffer_B[f][j+i*Number_of_ANDgates_per_PE]
                if(bitmap_buffer_B[f][j+i*Number_of_ANDgates_per_PE] != 0):
                    non_zero_temp_B.append(non_zero_buffer_B[non_zero_count_B])
                    non_zero_count_B = non_zero_count_B + 1
            non_zero_B_divisons_per_col.append(non_zero_temp_B)


    non_zero_temp_A_PE = []
    non_zero_temp_B_PE = []
    counter_PE_A = 0
    counter_PE_B = 0
    average_PE_util = 0
    average_cycle_count = 0
    max_PE_util = 0
    max_cycle_count =0 

    #assigning non-zeroes to their corresponding PEs
    for i in non_zero_A_divisons_per_col:
        non_zero_temp_A_PE.append(i)
        counter_PE_A = counter_PE_A + 1
        if counter_PE_A >= Number_of_ANDgates_per_PE:
            non_zero_A_per_PE.append(non_zero_temp_A_PE)
            non_zero_temp_A_PE = []
            counter_PE_A = 0

    for i in non_zero_B_divisons_per_col:
        non_zero_temp_B_PE.append(i)
        counter_PE_B = counter_PE_B + 1
        if counter_PE_B >= Number_of_ANDgates_per_PE:
            non_zero_B_per_PE.append(non_zero_temp_B_PE)
            non_zero_temp_B_PE = []
            counter_PE_B = 0



    #non_zero_A_per_PE, non_zero_B_per_PE provides insight which non- zeroes should go where
    #Bitmap_A_per_PE, Bitmap_B_per_PE provides insight about what are the bitmap values should go where
    #Now we should assign these values to the respective PEs and request the complete status 

   


    while (req_PEs!=0):
        print("req_PEs",req_PEs)
        
        if (req_PEs <= total_PEs):
            print ("I entered into req_PEs<=total_PEs loop")
            with Pool() as pool:
                args_list = [(buffer_A_per_PE[i], buffer_B_per_PE[i], non_zero_A_per_PE[i], non_zero_B_per_PE[i]) for i in range(req_PEs)]

                results = pool.starmap(Processing_element,args_list)

                pool.close()
                pool.join()

            req_PEs = 0 

        if(req_PEs >= total_PEs):
            print ("I didnot entered into req_PEs<=total_PEs loop")
        
            req_PEs_temp = req_PEs - total_PEs   
            with Pool() as pool:
                args_list = [(buffer_A_per_PE[i], buffer_B_per_PE[i], non_zero_A_per_PE[i], non_zero_B_per_PE[i]) for i in range(req_PEs_temp)]

                results = pool.starmap(Processing_element,args_list)

                pool.close()
                pool.join()

               ## print("cycle_count,PE_utilization",results)
            req_PEs = req_PEs - total_PEs

    print("cycle_count,PE_utilization", results)
    for i in range(len(results)):
        average_PE_util = average_PE_util + results[i][1]
    for i in range(len(results)):
        average_cycle_count = average_cycle_count + results[i][0]

    for i in range(len(results)):
        if (results[i][1]>max_PE_util):
            max_PE_util = results[i][1]
    min_PE_util = results[0][1]
    for i in range(len(results)):
        if (results[i][1]<min_PE_util):
            min_PE_util = results[i][1]

    
    for i in range(len(results)):
        if (results[i][0]>max_cycle_count):
            max_cycle_count = results[i][0]
    min_cycle_count = results[0][0]
    for i in range(len(results)):
        if (results[i][0]<min_cycle_count):
            min_cycle_count = results[i][0]


    print("average_PE_util",average_PE_util/len(results))
    print("maximum_PE_util",max_PE_util)
    print("minimum_PE_util",min_PE_util)
    print("minimum cycle count",min_cycle_count)
    print("maximum cycle count",max_cycle_count)
    print("average cycle_count",average_cycle_count/len(results))
    print("average_Mul_utilization/cycle_count",average_PE_util/average_cycle_count)

    

    """
    #this loop can be executed in parallel in HLS
    while(req_PEs_counter !=0):
        while(filled_PEs <= available_PEs and filled_PEs != req_PEs):
            if(req_PEs <= available_PEs):
                print("buffer_A_per_PE[filled_PEs],buffer_B_per_PE[filled_PEs],non_zero_A_per_PE[filled_PEs],non_zero_B_per_PE[filled_PEs]",buffer_A_per_PE[filled_PEs],buffer_B_per_PE[filled_PEs],non_zero_A_per_PE[filled_PEs],non_zero_B_per_PE[filled_PEs])
                status = Processing_element(buffer_A_per_PE[filled_PEs],buffer_B_per_PE[filled_PEs],non_zero_A_per_PE[filled_PEs],non_zero_B_per_PE[filled_PEs])
                completion_status.append(status)
                print("completion_Status",completion_status)
                req_PEs_counter = req_PEs_counter-1
                filled_PEs = filled_PEs+1
                print("filled_PEs,req_PEs_counter",filled_PEs,req_PEs_counter)
            else:
                req_PEs_counter = available_PEs
                available_PEs = available_PEs - req_PEs
        for i in range(req_PEs):
            if(completion_status[i] == 1):
                filled_PEs = filled_PEs-1 
    """                 

#Matrices generation for the input. You can specify the dimension of the matrix as well as density of the matrix
#rng = default_rng()
#rvs = stats.poisson(25, loc=10).rvs
#S2 = random(5,5, density=0.8, random_state=rng, data_rvs=rvs)
#S1 = random(512,512,density=0.8, random_state=rng,data_rvs=rvs)

#print("S2/n",S2)
S0 = np.loadtxt('random_split_0.txt').astype(int)
A = S0
B =  S0
#print("S0/n",S0)




#bitmap generator
def bitmap_generator(A):
    rows= len(A)
    columns = len(A[0])
    Bitmap_A = [[0]*columns for i in range(rows)]
    non_zero_A = []

    #for bitmap conversion
    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] != 0:
                Bitmap_A[i][j] = 1

            else:
                Bitmap_A[i][j] = 0

    #for non zero in bitmap conversion
    for j in range(len(A[i])):
        for i in range(len(A)):
            if A[i][j] != 0:
                non_zero_A.append(A[i][j])

    
    return non_zero_A, Bitmap_A

def Transpose(B):
    rows = len(B)
    columns = len(B[0])
    Col_stor = [[0]*rows for i in range(columns)]
    for j in range(columns):
        for i in range(rows):
            Col_stor[j][i] = B[i][j]
    
    return Col_stor

non_zero_A, Bitmap_A =bitmap_generator(A)
Column_storing = Transpose(B)

non_zeroes_A, bitmap_A = bitmap_generator(A)
Transpose_B = Transpose(B)
non_zeroes_B, bitmap_B = bitmap_generator(B)




if __name__ == '__main__':
    PE_scheduler (bitmap_A,bitmap_B,non_zeroes_A,non_zeroes_B,Number_of_ANDgates_per_PE)
