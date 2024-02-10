import random
# Define a function to read the file and process the data
def process_file(filename):
    data = [[None]]  # Initialize an empty list to store the processed data
    with open(filename, 'r') as file:
        for line in file:
            line_data = line.strip().split()  # Split each line into pairs of numbers
            tuples = [(int(line_data[i]), int(line_data[i+1])) for i in range(0, len(line_data), 2)]
            data.append(tuples)  # Append the list of tuples to the main data list
    return data

# Example usage
filename = 'jss_data.txt'  # Change 'data.txt' to the name of your input file
result = process_file(filename)
# print(result)  # Print the processed data

for i in result[1:]:
    i.append(0)

# last index of each list showing the 

machine_dict = {}

# Iterate through columns and dictionary mantained rn 
for col_idx in range(len(result[1]) - 1):  # Exclude the last element
    for row_idx, row in enumerate(result[1:], start=1):
        if row[col_idx] != 0 and row[col_idx] is not None:
            key = row[col_idx][0]
            value = [row_idx, row[col_idx][1]]
            if key in machine_dict:
                machine_dict[key].append(value)
            else:
                machine_dict[key] = [value]
# print(machine_dict)

print(result)
print(machine_dict)

timing_dict = {}
status_list = [True] * 10

def set_time(mdt, mac, job, end_time):
    # print("Is the end time for the job",end_time)
    # print("the job number to be checked is ", job)
    job_num = job
    current_index = result[job_num][-1]
    time_interval = result[job_num][current_index][1]

    # setting the index as well as this would indicate a jobu being endedu
    result[job_num][-1] +=1

    # if result[job_num][current_index] == len(result[job_num])-1:
    #     print("one job has been completed")
    #     status_list[job_num-1] = False
    #     print(status_list)


    if mac not in timing_dict:
        timing_dict[mac] = []
        timing_dict[mac].append((job_num,end_time - time_interval, end_time))
    else:
        timing_dict[mac].append((job_num,end_time - time_interval, end_time))



# calculating the fitness of the dictionary
# 1 element of the dictionary would never be 
def fitness(mcdt):
    time = 0
    while True in status_list and time < 100:
        time+=1
        # print("inside the while loop")

        # deducting the time and checking the aspect of 
        for numb in machine_dict:
            # print("inside the for loop")

            if machine_dict[numb] == []:
                status_list[numb-1] = False
                continue
            job_numb = machine_dict[numb][0][0] # get the job number which is currently executed on the machine
            current_machine_job = result[job_numb][-1]
            machine_order = result[job_numb][current_machine_job][0]

            # this code ensures that the order is mantained as in the
            # correspondence with the order that is needed to be executed
            # thus this would lead the machine to wait
            if machine_order!= numb:
                continue
            
            machine_dict[numb][0][1] -= 1
            if machine_dict[numb][0][1] == 0:
                # popping the job which has been completed
                pop_job_numb = machine_dict[numb].pop(0)   
                pop_job_numb = pop_job_numb[0]
                # print("job which has been completed is ", pop_job_numb)
                # seeting the time number and updating all the relevant
                # dictionaries and pointer of the list
                set_time(machine_dict, numb, pop_job_numb, time)
    print(time)

for i in machine_dict:
    print("machine number is ", i , "  with the first element ", machine_dict[i][0])

fitness(machine_dict)

print(machine_dict)
print(status_list)
print(timing_dict)

for i in machine_dict:
    print("machine number is ", i , "  with the first element ", machine_dict[i][0])



# print(status_list)

                



                  






                




            

        


    






# for i in result:


# supposing the number of jobs to be 10 for now
# jobList = [] * 10
# class entity():