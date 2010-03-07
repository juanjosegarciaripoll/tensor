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
#report {
font-size: 12px;
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
        </style>
      </head>
      <body>
        <h1>Configuration log</h1>
        <table id="report">
          <tbody>
            <tr>
              <th colspan="2">Configuration</th>
            </tr>
            <xsl:for-each select="testframe/config">
              <tr>
              <td><xsl:value-of select="@field"/></td>
              <td><xsl:value-of select="@value"/></td>
              </tr>
            </xsl:for-each>
          </tbody>
        </table>
        <table id="report">
          <tbody>
            <xsl:for-each select="testframe/testsuites">
              <tr>
                <td colspan="2" class="framework"><xsl:value-of select="@name"/></td>
              </tr>
              <xsl:apply-templates/>
            </xsl:for-each>
            <xsl:for-each select="testframe/testsuite">
              <tr>
                <td colspan="2" class="framework"><xsl:value-of select="@name"/></td>
              </tr>
              <xsl:apply-templates/>
            </xsl:for-each>
          </tbody>
        </table>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="globalfailure">
    <tr>
      <td colspan="2">
        <xsl:value-of select="@name"/>
      </td>
    </tr>
    <tr>
      <td></td>
      <td class="failed">FAILED</td>
    </tr>
  </xsl:template>

  <xsl:template match="testsuite">
    <tr>
      <td colspan="2" class="suite">
        <xsl:value-of select="@name"/>
      </td>
    </tr>
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="testcase">
    <tr>
      <td class="case"><xsl:value-of select="@name"/></td>
      <xsl:choose>
        <xsl:when test="failure"><td class="failed">FAILED</td></xsl:when>
        <xsl:otherwise><td class="ok">OK</td></xsl:otherwise>
      </xsl:choose>
    </tr>
    <xsl:for-each select="failure">
      <tr>
        <td colspan="2" class="failure"><xsl:value-of select="@message"/>
        </td>
      </tr>
    </xsl:for-each>
  </xsl:template>

</xsl:stylesheet>
