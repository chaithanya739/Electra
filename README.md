# Electra
Electra: Eliminating the Ineffectual Computations of Bitmap Compressed Matrices

This is a Python based cycle accurate simulator of Electra. The performance of the model determined through the use of software counter (cycle count). The cycle time of the software counter is equal to the operation time of logical AND. 

The parameters can be adjusted in accordance with the user inputs that determines the hardware resources of the simulator. The number of multiplier and adder counters are selected in selected in such a way that atleast one multiplier and adder counters should be available in each cycle to remove the requirement of queues in PEs. This restriction provides better utilization of multiplier and adder counters.

The number of mulipliers, adders and logical elements are modeled with the help of classes. The operation of low level scheduler is implemented in the scheduler function. Processing_element provides the implementation of processing element which consists of logical element operation, low level scheduler operation, multiplier and adder counters, merger unit to merge the results while flushing out 
