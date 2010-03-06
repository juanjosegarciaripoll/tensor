<?xml version="1.0"?>
<!DOCTYPE xsl:stylesheet [
<!ENTITY nbsp "&#160;">
]>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="html" indent="yes"/>
  
  <xsl:template match="/">
    <html>
      <head>
        <title>o:XML Test Report Interface</title>
      </head>
      <body>
        <table>
          <tbody>
            <xsl:apply-templates/>
          </tbody>
        </table>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="testsuite">
    <xsl:if test="not(@name='AllTests')">
      <tr>
        <td colspan="2">
          <xsl:value-of select="@name"/>
        </td>
      </tr>
    </xsl:if>
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="testcase">
    <tr>
      <td><xsl:value-of select="@name"/></td>
      <td>
        <xsl:choose>
          <xsl:when test="failure">FAILED</xsl:when>
          <xsl:otherwise>OK</xsl:otherwise>
        </xsl:choose>
      </td>
    </tr>
    <xsl:for-each select="failure">
      <tr>
        <td colspan="2"><xsl:value-of select="@message"/>
        </td>
      </tr>
    </xsl:for-each>
  </xsl:template>

  <xsl:template match="failure">
    <tr>
      <td colspan="2" class="failure">
        <xsl:value-of select="@message" />
      </td>
    </tr>
  </xsl:template>
</xsl:stylesheet>
