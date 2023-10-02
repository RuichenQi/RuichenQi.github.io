---
layout: post
title: Overview of the Development of Electronic Design Automation(EDA) Industry
subtitle: This article contains an overview of the development of EDA industry, including the development of EDA tools, the main contents of EDA and the development trends of EDA.
categories: EDA
tags: EDA_Tools
---
# Overview of the Development of Electronic Design Automation(EDA) Industry

## Introduction

After the 20th century, with the rapid development of the electronics industry, especially integrated circuit technology, traditional electronic circuit design methods no longer meet the requirements of industrial development. Electronic Design Automation (EDA) is a design method that uses computers as the carrier, and designers use Hardware Description Language (HDL) to complete design files on the EDA software platform. It is the computer that automatically completes compilation, synthesis, optimization, layout and routing, simulation, adaptation compilation, logical mapping and programming download for specific target chips. EDA technology is a modern electronic discipline that has developed rapidly in recent years which applies computer technology, software technology, electronic system and microelectronics technology. EDA technology has become the core of modern electronic technology design.

## The Development of Electronic Design Automation

Electronic design automation refers to the use of computer-aided design software to complete the design of functionality, synthesis, verification, layout, routing and design rule inspection of VLSI chips. The basic characteristic of EDA technology is that EDA tools can transform abstract behavior-level circuits described by hardware description languages into actual circuit structures, with system-level simulation and synthesis capabilities. Before the advent of electronic design automation technology, designers had to manually complete the design, layout and routing of integrated circuits. With the rapid development of the integrated circuit industry, the scale and structural complexity of circuits have drastically increased, which requires automating the entire integrated circuit design process. The development of electronic design automation can be divided into three stages[1]:

### 1. Computer Aided Design(CAD) phase

In the 70s of the 20th century, integrated circuits entered the era of Complementary Metal Oxide Semiconductor (CMOS). This stage is the beginning of the development of EDA technology. People began to use computer software for IC and PCB layout and routing. The introduction of computer aided design technology frees designers from heavy workload of layout and routing. At this stage, limited by the performance and design platform, the design work was not well supported, the level of automation was low, and many tasks still needed to be done manually.

### 2. Computer Aided Engineering(CAE) phase

In the 80s of the 20th century, people began to use computers to complete logic simulation, timing analysis, automatic layout and routing. However, in CAD/CAE phase, the bottom-up design approach cannot guarantee the first-time success of the design, thereby extending the development cycle to a certain extent[2]. Electronic design automation tools at this stage still have great limitations and cannot well meet the complex requirements required for electronic design.

### 3. Electronic design automation(EDA) phase

Since the 90s of the 20th century, EDA tools have not only helped designers complete electronic system design, but also provided system-level design and simulation independent of process and manufacturer. After decades of continuous development, EDA technology has become a popular technology that can be independently designed and simulated, which directly promotes the development of the electronic information industry. EDA technology has truly entered the golden age at this time.

## The main content of electronic design automation

### 1. Hardware description language

Hardware description language is a language used for hardware behavior description, structure description, and data flow description of electronic systems. At present, any EDA tool is inseparable from HDL[2]. There are two mainstream hardware description languages: Verilog HDL and VHDL.

Verilog was founded in late 1983 by Phil Moorby, an engineer at Gateway Corp. In 1990, Gateway Corp was acquired by Cadence. In 1992, Open Verilog International (OVI) included Verilog in the Institute of Electrical and Electronics Engineers standard. After several subsequent updates, in 2009, the IEEE 1364-2005 standard and the IEEE 1800-2005 standard were merged into the IEEE 1800-2009 standard, which is now the popular SystemVerilog.

VHDL was originally developed by the U.S. Department of Defense as a design language for the U.S. military to improve design reliability and shorten development cycles. In 1987, VHDL was recognized by the U.S. Department of Defense and IEEE as the standard hardware description language. In 1993, IEEE revised VHDL to form the IEEE 1076-1993 standard. At this point, VHDL became the factual general-purpose hardware description language.

What Verilog HDL and VHDL have in common is that they can both formally abstract the behavior and structure of a circuit; Support the description of hierarchy and scope in logical design; The description of circuit behavior can be simplified; With circuit simulation and verification mechanism; Support comprehensive conversion of circuit description from high level to low layer; Regardless of the implementation process; Easy to manage and design reuse. The difference between Verilog HDL and VHDL is that Verilog HDL was launched earlier, so it has more resources and a larger customer base; Verilog HDL syntax is close to C programming language and easier to master by beginners than VHDL; Conventional wisdom holds that Verilog HDL is weak in terms of system-level abstraction. However, after the supplement of the Verilog 2001 standard, the system-level abstraction and synthesizability of Verilog HDL have been greatly improved[3].

### 2. Software development tools

EDA tool software can be roughly divided into three categories: chip design auxiliary software, programmable chip auxiliary design software, and system design auxiliary software. EDA software occupies an important position in the EDA system. Using EDA software, electronic designers can design electronic systems from concepts, algorithms, protocols, etc., and a lot of work can be done by computers, and electronic products can be automatically processed on the computer from circuit design, performance analysis to the whole process of designing IC layout or PCB layout. IC design tools have evolved rapidly since their inception and have themselves evolved to the ASIC chip design stage. The three most famous EDA software companies in the world are Synopsys, Cadence in the United States and Mentor Graphics, a subsidiary of Siemens.

In the mid-to-late eighties of last century, China began to invest in EDA industry research and development. In 1993, the first EDA tool called Panda System produced in China came out. After that, the development of EDA in China was tortuous and slow. At present, China's EDA software market is basically monopolized by foreign companies. China's EDA indstry is short of talents and the industrial chain is broken; Limited by the high threshold of EDA technology, the development of independent EDA tool in China is slow, and almost all core technologies are mastered by foreign companies. The development of EDA software in China still has a long way to go.

### 3. Programmable logic devices

Field Programmable Gate Arrays(FPGAs) and Complex Programmable Logic Devices(CPLDs) are Programable Logic Devices(PLDs). FPGAs and CPLDs have a larger scale than previous simple logic cells. Since EDA software has been quite developed, users can even complete the design of fairly excellent circuits or systems without detailed understanding of the internal structure or principles of programmable logic devices. The circuit or system designed by programmable logic devices has the advantages of high integration, short development cycle and low development difficulty, and replacing traditional standard integrated circuits with programmable logic devices has become the trend of digital technology development[4].

### 4. Application-specific integrated circuits

Application Specific Integrated Circuit (ASIC) is a kind of integrated circuit designed for specialized purposes. ASIC is one of the most widely used specialized integrated circuits, with the advantages of high efficiency and good performance. The design patterns of ASICs can be divided into two types: fully customized design and semi customized design. Usually, fully customized design requires designers to complete the entire design of integrated circuits, resulting in a longer design cycle and higher design difficulty. However, compared to semi customized design, it has the advantages of faster operation speed and lower power consumption; Semi custom design refers to the designer introducing standard logic units and IP cores from the library to complete partial circuit design. Semi customized design can reduce design costs, significantly shorten design cycles, and save time expense. The design of ASIC cannot be separated from EDA technology. To complete the design and implementation of ASIC, EDA technology can be used for electronic system design and functional verification. Completing ASIC design through CPLD and FPGA is currently one of the most popular methods in both research and industry.

## The Development Trend of Electronic Design Automation

The rapid development of the electronic industry today has put forward new requirements for integrated circuit designers. Completing the design and manufacturing of integrated circuits quickly and effectively, shortening the research and development cycle as much as possible, and reducing research and development costs require more powerful EDA tools as a guarantee. At present, the development of the EDA industry mainly has the three following trends[5]:

### 1. Moving towards high-density, high-speed, and broadband

The development of the electronics industry is inseparable from the advancement of EDA technology. At present, high-density, high-speed, wide-band programmable logic devices have become the mainstream, and the development of these programmable logic devices has greatly promoted the development of the electronics industry. The progress of design methods and design efficiency has greatly promoted the development of devices and the progress of processes, and with each improvement of manufacturing processes, the integration, scale, and computing speed of programmable logic devices will be improved.

### 2. Moving towards predictable delays

Today's big data application scenarios require data storage at a larger scale and more advanced processing capabilities. With the gradual increase of the scale of data flow, the real-time requirements for data processing are getting higher and higher, which puts forward new requirements for the development of EDA technology in the future. Predictable latency of integrated circuits is very important to meet the efficient real-time performance required by large-scale data processing.

### 3. Moving towards low power consumption

According to Moore's Law, when the price is constant, the number of components that can be accommodated on an integrated circuit doubles approximately every 18-24 months. With the increase of the scale of integrated circuits, circuit power consumption has become one of the important factors restricting the development of integrated circuits. How to reduce power consumption and improve the computing efficiency of circuits is one of the trends in the development of EDA technology in the future.

## Summary

The emergence of EDA technology is a major technological revolution in the field of electronic design and the electronic industry, which has greatly improved the efficiency and flexibility of circuit design. EDA technology has been widely used in electronic technology, computer technology, signal processing, biomedical, military equipment and other fields. Companies such as Altera(now merged into Intel), Xilinx(now merged into AMD) and Siemens have achieved strong technical accumulation in this field. China's EDA industry has developed rapidly in recent years, but there is still a large technological gap compared with developed countries. As one of the most dynamic and promising technologies in the field of electronic design, EDA technology is a promising technology with broad application prospect.

## References

[1]乔序.EDA技术发展与应用[J].大观周刊,2012(18):108-109.  
[2]杨焯群.EDA技术发展综述[J].电子制作,2018(1):90-91.  
[3]徐文波,田耘.Xilinx FPGA开发实用教程(第二版).  
[4]徐晓峰,李鹏.可编程逻辑器件(PLD)的发展及VHDL语言的应用[J].煤炭科技,2002(2):37-38.  
[5]钱刚.电子设计自动化的发展研究[J].无线互联科技,2017(11):68-69.  
