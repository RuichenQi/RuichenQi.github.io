---
layout: post
title: A 20-minute guide to writing simple but elegant documents using Markdown
subtitle: This is a simplified tutorial including the most commonly-used markdown syntax that can help you write documents efficiently.
categories: Markdown
tags: Guide
---

# A 20-minute guide to writing simple but elegant documents using Markdown

---

*Preface: Markdown is a simple syntax that formats text as headers, lists, boldface, and so on. It is popular among people who write for the web, as it is easy to read and write. Markdown can be used to create websites, documents, notes, books, presentations, email messages, and technical documentation.*

*HTML in Markdown is a way to use HTML tags in Markdown-formatted text. This is helpful if you prefer certain HTML tags to Markdown syntax, or if you need to use some HTML features that are not supported by Markdown. In this guide, we use several uncomplicated HTML tags to achieve some HTML features that are not well-supported by Markdown itself. It is worth mentioning that **you should not use HTML tags that affect the document structure or rely on external resources, such as \<head>, \<body>, \<title>, etc., as they might conflict with the output of the Markdown processor.***

*This guide covers all the basic and most-commonly used Markdown syntax knowledge you need to write a document, to help you get started with editing documents using Markdown. For the rest of the parts that are not covered in this document, please refer to [Markdown Guide](https://www.markdownguide.org/) and [Markdown Wikipedia](https://en.wikipedia.org/wiki/Markdown).*

---

## Headers
Headers are used in an article to organize the content and make it easier for the reader to follow. Headers are usually written in a larger font and in bold to stand out from the rest of the text. There are different levels of headers, such as H1, H2, H3, etc., that indicate the hierarchy of the sections and subsections.

You can add Headers using '#' with a space before your header.
```Markdown
# Header H1
## Header H2
### Header H3
#### Header H4
##### Header H5
###### Header H6
```
And the results will look like this:
# Header H1
## Header H2
### Header H3
#### Header H4
##### Header H5
###### Header H6

---

## Lines
There are two ways for you to break lines. You can break lines by appending two spaces to the end of the line. You can also use the HTML-style \<br> tags to represent a line break.
```Markdown
First line with two spaces after.  
And the next line.

This is your first line.<br> This is your second line.
```
The result is shown below:

First line with two spaces after.  
And the next line.

This is your first line.<br> This is your second line.

---

## Paragraphs
You can simply write in a new line to restart a paragraph, or you can create a new paragraph using the synax below.
```Markdown
This is your first paragraph. <p>Your next paragraph here.</p>
```
The result is as follows:

This is your first paragraph. <p>Your next paragraph here.</p>

---

## Font formats
Font formats roughly include typeface, font size, font color, bold, italic, underline, strikethrough and so on. Since markdown does not support some of the features in format changing, we can implement these features using embedded HTML code.
```Markdown
<font face="Times New Roman">Times New Roman</font>

<font size=5px>size=5px</font>

<font color=#FF0000 >Red</font>

**bold text**

*Italic*

<u>underline</u>

~~strikethrough~~
```
The results will look like this:

<font face="Times New Roman">Times New Roman</font>

<font size=5px>size=5px</font>

<font color=#FF0000 >Red</font>

**bold text**

*Italic*

<u>underline</u>

~~strikethrough~~

Besides, you can also combine some of these parameters to achieve a better performance.
```Markdown
*<font face="Times New Roman" size=5px color=#00FFFF><u>This is an example of combination.</u></font>*
```
It looks like this:

*<font face="Times New Roman" size=5px color=#00FFFF><u>This is an example of combination.</u></font>*

---

## Block quotes
A block quote is a way of formatting a long quotation in a written document, such as an essay or a report. A block quote is set off from the main text as a separate paragraph or block of text, and usually has a different font size or style.

To create a block quote, add a '>' symbol before the paragraph. For example:
```Markdown
>This is a block quote.
```
This is how it will look like:
>This is a block quote.

You can also add a '>>' symbol after the block quote to achieve sub block quote. For example:
```Markdown
>This is a block quote.
>>This is a sub block quote
```
This is how it will look like:
>This is a block quote.
>>This is a sub block quote

---

## Lists
A list in Markdown is a way of formatting a series of items or tasks in a document. You can create different types of lists in Markdown, such as unordered lists, ordered lists, and description lists.

To create an unordered list or a bullet list, you can use asterisks (*), hyphens (-), or pluses (+) to mark each item. For example:
```Markdown
* Item1
- Item2
+ Item3
```
This will produce a list like this:
* Item1
- Item2
+ Item3

You can also nest unordered lists by adding more spaces or tabs before the markers. For example:
```Markdown
* Item 1
    * Sub-item 1
    * Sub-item 2
* Item 2
    - Sub-item 3
    - Sub-item 4
```
This will produce a list like this:
* Item 1
    * Sub-item 1
    * Sub-item 2
* Item 2
    - Sub-item 3
    - Sub-item 4

Creating an ordered list is much similar to creating unordered list. The only difference is exchanging asterisks (*), hyphens (-), or pluses (+) with order numbers. For instance:
```Markdown
1. Item 1
    1. Sub-item 1
    2. Sub-item 2
2. Item 2
    1. Sub-item 3
    2. Sub-item 4
```
This will produce a list like this:
1. Item 1
    1. Sub-item 1
    2. Sub-item 2
2. Item 2
    1. Sub-item 3
    2. Sub-item 4

---
## Tables
You can use vertical bars (|) to separate each column, and use three or more dashes (-) to create headers for each column. You can use colons (:) to adjust column alignment. Add a colon to the left of the dash for left alignment, a colon to the right for right alignment, and colons on both sides for center alignment.

Here comes an example:
```Markdown
| Number | Next number | Previous number |
| :- |:- |:- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |
```
It will look like:

| Number | Next number | Previous number |
| :- |:- |:- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |

---

## Adding images or hyperlinks
To add an image, you need to start with an exclamation mark (!), followed by the alternative text for the image in square brackets ([]), and then the URL or path of the image in parentheses (()). For example:
```Markdown
Now you can see an image of Klee:

![Klee](../assets/images/markdown/klee.jpg)
```

The result is:

Now you can see an image of Klee:

![Klee](./RuichenQi.github.io/tree/main/assets/images/markdown/Klee.jpg)

To resize or move the image, create a \<div> element that is aligned to the center of the page, and contains an \<img> element that displays an image of Klee: 
```Markdown
<div  align="center">  
 <img src="../assets/images/markdown/klee.jpg" width = "150" height = "200" alt="Klee" align=center />
</div>
```
Now you can see a resized and centered image of Klee:
<div  align="center">  
 <img src="../assets/images/markdown/klee.jpg" width = "150" height = "200" alt="Klee" align=center />
</div>
You can also add images or website addresses as hyperlinks:

```Markdown
[Klee](../assets/images/markdown/klee.jpg)
[Visit my home page](https://ruichenqi.github.io/)
```
And now you can view the image or browse the website by clicking hyperlinks:

[Klee](../assets/images/markdown/klee.jpg)

[Visit my home page](https://ruichenqi.github.io/)

---

## Code blocks
Nothing is more thrilled than sharing your code with others! Code blocks are a way of formatting and displaying a piece of code in a document. Code blocks can help you avoid plagiarism, show that you have done research, provide evidence and credibility for your claims, and allow your readers to find and verify your sources.

To insert inline code, use the form \`your code here` to insert it.

To insert multiple lines of code, use three backticks (```) to wrap the multiple lines of code. Or use indentation.

For example:
````Markdown
`print("Hello, Python!")`

```Cpp
void main(){
  std::cout << "Hello, Cpp!" << std::endl;
}
```
````
The result is shown as follows:

`print("Hello, Python!")`

```Cpp
void main(){
  std::cout << "Hello, Cpp!" << std::endl;
}
```

---

## Escape characters
To display characters originally used to format a Markdown document, precede the characters with a backslash character \

The characters listed below can be escaped using the backslash character:
| Character | Name | Character | Name |
| :- |:- | :- |:- |
| \ | backslash | # | pound sign |
| ` | backtick | + | plus sign |
| * | asterisk | - | minus sign (hyphen) |
| _ | underscore | . | dot |
| {} | curly braces | ! | exclamation mark |
| [] | brackets | \| | pipe |
| () | parentheses |

---

## Formulas and equations
To insert mathematical formulas and equations in Markdown, you can use LaTeX style syntax to render math expressions within Markdown inline (using $ delimiters) or in blocks (using $$ delimiters). Here is an example:
```Markdown
Quadratic function: $y=ax^2+bx+c$

Euler's formula (One of the most elegent formulas in the world):
$$e^{ix}= (cos x+isin x)$$
```

Then you can see the beautiful mathematical formulas:

Quadratic function: $y=ax^2+bx+c$

Euler's formula (One of the most elegent formulas in the world):
$$e^{ix}= (cos x+isin x)$$

---

## Separators
To create a separator, use three or more asterisks (***), dashes (---), or underscores (___) on a single line with no other content.
For compatibility reasons, please leave at least a blank line above and below the separator line.

---

## What are not included here?
- How to insert HTML code blocks
- How to embed streaming player or audio player, etc.
- More advanced manipulations and operations that may be helpful.

---

*Postscript: For security reasons, not all Markdown applications support adding HTML to Markdown documents. When in doubt, check the manual for the corresponding Markdown application. Some applications only support a subset of HTML tags.*

*For HTML block-level elements \<div>, \<table>, \<pre>, and \<p>, use blank lines before and after them to separate them from other content. Try not to use tabs or spaces to indent HTML tags, otherwise the format will be affected.*

*Markdown syntax cannot be used within HTML block-level tags. For example \<p>italic and \*\*bold\*\*\</p> will not work.*
