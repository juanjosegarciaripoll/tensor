<?xml version="1.0"?>
<!DOCTYPE xsl:stylesheet [
<!ENTITY nbsp "&#160;">
]>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="html" indent="yes"/>
  
  <xsl:template match="/">
    <html>
      <head>
        <title>Tensor C++ Library -- Test results</title>
        <style type="text/css">
html {
font-family: sans serif, Arial;
}
* {
font-size: inherit;
line-height: inherit;
font-family: inherit;
}
table {
padding: 0;
}
body {
max-width: 65em;
font-size: 12px;
line-height: 1.5em;
}
#report {
text-align: left;
}
.ok {
color: green;
}
.failed {
color:red;
}
.framework {
background-color: #CCC;
}
.suite {
padding-left: 1em;
}
.case {
padding-left: 2em;
}
.failure {
padding-left: 4em;
color: red;
}
.left {
display: inline-block;
width: 80%;
}
.right {
display: inline-block;
width: 10%;
}
.full {
display: block;
width: 100%;
}
        </style>
      </head>
      <body>
        <table id="report">
          <tbody>
            <tr>
              <th colspan="2">Configuration <a href="../test.html">[back]</a></th>
            </tr>
            <xsl:for-each select="testframe/config">
              <tr>
              <td><xsl:value-of select="@field"/></td>
              <td><xsl:value-of select="@value"/></td>
              </tr>
            </xsl:for-each>
          </tbody>
        </table>
        <div id="report">
	  <xsl:for-each select="testframe/globalfailure">
	    <div class="framework full"><xsl:value-of select="@name"/></div>
	    <div class="suite left"></div>
	    <div class="failure right">Failed</div>
	  </xsl:for-each>
	  <xsl:for-each select="testframe/testsuites">
	    <div class="framework full"><xsl:value-of select="@name"/></div>
	    <xsl:apply-templates/>
	  </xsl:for-each>
	  <xsl:for-each select="testframe/testsuite">
	    <div class="framework full"><xsl:value-of select="@name"/></div>
	    <xsl:apply-templates/>
	  </xsl:for-each>
        </div>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="globalfailure">
    <div class="framework full"><xsl:value-of select="@name"/></div>
    <div class="suite left"></div>
    <div class="failure right">Failed</div>
  </xsl:template>

  <xsl:template match="testsuite">
    <div class="suite full"><xsl:value-of select="@name"/></div>
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="testcase">
    <div class="case left"><xsl:value-of select="@name"/></div>
    <div class="right">
      <xsl:choose>
	<xsl:when test="failure"><div class="failed">FAILED</div></xsl:when>
	<xsl:otherwise><div class="ok">OK</div></xsl:otherwise>
      </xsl:choose>
    </div>
    <xsl:for-each select="failure">
      <div class="failure full">
	<xsl:value-of select="@message"/>
      </div>
    </xsl:for-each>
  </xsl:template>

</xsl:stylesheet>
