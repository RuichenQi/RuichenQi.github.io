---
layout: post
title: Introduction to High Level Synthesis and FPGA-based Neural Network Accelerator.md
subtitle: This article introduces the basic concepts of electronic design automation (EDA), hardware description language (HDL) and high-level synthesis (HLS), introduces the basic process of high-level synthesis and several ideas of neural network hardware accelerator design using one of the mainstream HLS tools - Vivado HLS.
categories: EDA
tags: EDA
---

# Introduction to High-level Synthesis

## Electronic Design Automation (EDA)

Electronic design automation refers to the design method that uses computer-aided design software to complete the functional design, synthesis, verification, layout, routing, design rule checking and other processes of integrated circuit. The basic characteristics of EDA technology are the use of hardware description language to describe circuits and support system-level simulation and synthesis. The development of electronic design automation can be divided into three stages: computer-aided design stage, computer-aided engineering design stage and electronic design automation stage. Electronic design automation tool software can be roughly divided into three categories: chip design auxiliary software, programmable chip auxiliary design software and system design auxiliary software. EDA software occupies an important position in the EDA industry. Using EDA software, electronic designers can design electronic systems starting from concepts, algorithms, protocols, etc. A large amount of work can be completed through computers, including the entire process of electronic products design from circuit design and verification to IC or PCB layout and routing to performance analysis. Processing is completed automatically on the computer. EDA tools can not only help designers complete electronic system design, but also provide system-level design and simulation independent of processes and manufacturers.

## Hardware Description Language (HDL)

Hardware description language is a kind of programming language used to describe the behavior, structure, and data flow of hardware system. At present, any EDA tool is inseparable from hardware description language. There are currently two mainstream hardware description languages: Verilog HDL and VHDL. What they have in common is that they can formally and abstractly represent the behavior and structure of the circuit; support the description of levels and scopes in logic design; can simplify the description of circuit behavior; have circuit simulation and verification mechanisms; support circuit description by high-level comprehensive conversion to low-level; independent of implementation process; easy for management and design reuse. But they also have certain shortcomings, such as being difficult to understand, taking a long time to get started, and being unfriendly to software developers. In other words, it is difficult for traditional software engineers to use Verilog or VHDL languages to implement complicated algorithms on FPGA.

## High-level Synthesis (HLS)

High-level synthesis is deesigned to solve the above problems. HLS is a code synthesis tool that can convert logic, algorithms and functions described by high-level programming languages such as C/C++ into register transfer level (RTL) circuits described by hardware programming languages such as Verilog and VHDL, which not only improves the development speed, reduces the difficulty of development, and shortens the development cycle, but more importantly, allows software engineers to easily achieve complicated algorithms based on FPGA and other devices with high parallel processing capabilities.

## What is Vivado HLS

Vivado High-Level Synthesis (HLS) is a tool developed by Xilinx that allows you to compile and execute your C, C++, or SystemC algorithm, synthesize the C design to an RTL (Register Transfer Level) implementation, review the reports, and understand the output file. This tool is part of the Vivado Design Suite.

In addition to compatibility and optimization of commonly used C/C++ standard libraries, Vivado HLS also adds optimized libraries for image and video processing algorithms. What's more, the available data types in Vivado HLS are rich, and fixed-point data types can also be defined through related libraries. More importantly, in order to better support the development of IP cores, Vivado HLS also incorporates a variety of hardware interface encapsulation, data encapsulation and optimization instructions. Therefore, compared to the development methods of Verilog and VHDL, Vivado HLS greatly reduces the difficulty of implementing algorithms and functions on FPGA especially for the hardware design and implementation of complex algorithms such as image processing, video stream processing, and machine learning algorithms on FPGA.

## Development Process Using HLS Tools

*Step 1:* Use the data types, standard libraries and additional libraries provided by HLS tools to write the C/C++ code required to implement logic, algorithms or functions, and write the TestBench script for testing. 

*Step 2:* Use the TestBench test script Carry out C functional simulation with algorithm C/C++ code to verify the functional correctness of the algorithm.  

*Step 3:* Use the control constraints, interface encapsulation and optimization instructions provided by Vivado HLS to constrain the algorithm code verified through C functional simulation. Encapsulation and optimization, including parallel optimization and pipeline optimization.  

*Step 4:* Perform C synthesis on the optimized algorithm code, that is, HLS high-level synthesis, and convert it into RTL-level Verilog or VHDL code.  

*Step 5:* Combine with TestBench test script Perform RTL co-simulation on the high-level synthesized algorithm code, that is, C/C++ and RTL co-simulation, to ensure that the algorithm function and C function simulation results are consistent.  

*Step 6:* Package the algorithm code that has passed RTL co-simulation into a hardware IP package And output, the hardware algorithm IP can be later called in the Vivado design and development kit for system integration design.

## Four solutions of design of FPGA-based neural network hardware accelerator

The development of neural network hardware accelerator design solutions based on FPGA can be summarized into four types:

(1) implementing a complete neural network network model in FPGA.  

(2) dividing each neuron layer of the neural network into two parts: the software part and the hardware part. The formal part is implemented in the form of software on a software processor (such as ARM), and the latter part is implemented in the form of hardware in FPGA.  

(3) A part of the calculation of the neural network is implemented in the FPGA, and the complete neural network is implemented through the configuration and call of the hardware by the software.  

(4) Design and implement a general neural network hardware processor based on a specific instruction set in FPGA. 

The above four design solutions tradeoff speed, flexibility and FPGA hardware resource consumption in FPGA-based neural network hardware accelerator design from different perspectives. However, the FPGA hardware resource consumption of option (1) is large, the performance and flexibility of option (2) are poor, and for option (4), the cost, time requirement and difficulty of design and implement are all too high. Instead, option (3) balances speed, flexibility and FPGA hardware resource consumption to a certain extent, and is more suitable for designing embedded neural network hardware accelerator.
