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
#report td, #report th {
padding: 0 1em;
}
.ok {
color: green;
}
.failed {
color:red;
}
        </style>
      </head>
      <body>
        <table id="report">
          <tbody>
            <tr>
              <th colspan="7">Build farm brief report</th>
            </tr>
            <tr>
              <th>OS</th>
              <th>Vendor</th>
              <th>Architecture</th>
              <th>Libraries</th>
              <th>Link</th>
              <th>Failures</th>
              <th>Date</th>
            </tr>
            <xsl:apply-templates/>
          </tbody>
        </table>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="testframe">
    <tr>
    <td>
      <xsl:element name="a">
        <xsl:attribute name="href">logs/<xsl:value-of select="@name"/>.html</xsl:attribute>
        <xsl:for-each select="config">
          <xsl:if test="@field = 'host_os'">
            <xsl:value-of select="@value"/>
          </xsl:if>
        </xsl:for-each>
      </xsl:element>
    </td>
    <td>
      <xsl:for-each select="config">
        <xsl:if test="@field = 'host_vendor'">
          <xsl:value-of select="@value"/>
        </xsl:if>
      </xsl:for-each>
    </td>
    <td>
      <xsl:for-each select="config">
        <xsl:if test="@field = 'host_cpu'">
          <xsl:value-of select="@value"/>
        </xsl:if>
      </xsl:for-each>
    </td>
    <td>
      <xsl:for-each select="config">
        <xsl:choose>
        <xsl:when test="@field = 'TENSOR_USE_MKL' and @value = 1">MKL</xsl:when>
        <xsl:when test="@field = 'TENSOR_USE_ATLAS' and @value = 1">Atlas</xsl:when>
        <xsl:when test="@field = 'TENSOR_USE_VECLIB' and @value = 1">VecLib</xsl:when>
        </xsl:choose>
      </xsl:for-each>
    </td>
    <td>
      <xsl:for-each select="config">
        <xsl:choose>
        <xsl:when test="@field = 'SHARED' and @value = 'yes'">shared</xsl:when>
        <xsl:when test="@field = 'SHARED' and @value = 'no'">static</xsl:when>
        </xsl:choose>
      </xsl:for-each>
    </td>
    <xsl:choose>
      <xsl:when test="@failures=0">
        <td class="ok">0</td>
      </xsl:when>
      <xsl:otherwise>
        <td class="failed"><xsl:value-of select="@failures"/></td>
      </xsl:otherwise>
    </xsl:choose>
    <td>
      <xsl:for-each select="config">
        <xsl:if test="@field = 'date'">
          <xsl:value-of select="@value"/>
        </xsl:if>
      </xsl:for-each>
    </td>
    </tr>
  </xsl:template>

</xsl:stylesheet>