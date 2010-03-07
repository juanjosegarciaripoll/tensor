LOGS_DIR = $(HOME)/src/tensor_logs

www: test.html logs

test.html: $(top_srcdir)/scripts/joint_report.xsl joint_report.xml Makefile
	xsltproc $(top_srcdir)/scripts/joint_report.xsl joint_report.xml > $@

joint_report.xml: Makefile
	echo '<?xml version="1.0" encoding="UTF-8"?>' > $@
	echo '<testfarm>' >> $@
	for i in $(LOGS_DIR)/*/report.xml; do \
	  dir=`dirname $$i`; \
	  name=`basename $$dir`; \
	  nfail=`cat $${dir}/report_nfailures`; \
	  if [ "x$${nfail}" = "x" ]; then nfailures=1; fi; \
	  sed  -e "/?xml/d;s/<testframe>/<testframe name='$$name' failures='$$nfail'>/" $$i >> $@ ; \
	done
	echo '</testfarm>' >> $@

logs: $(LOGS_DIR) Makefile $(top_srcdir)/scripts/report.xsl
	if [ -d logs ]; then rm -f logs/*.html ; else mkdir logs; fi
	for i in $(LOGS_DIR)/*/report.xml; do \
	  dir=`dirname $$i`; \
	  name=`basename $${dir}`; \
	  xsltproc $(top_srcdir)/scripts/report.xsl $$i > logs/$${name}.html; \
	done

upload: test.html logs ../doc/html
	rsync --delete -rauvz ../doc/html test.html logs dreamhost:tream.dreamhosters.com/tensor/

../doc/html:
	make -C .. doxygen-doc

clean:
	rm -f *.html *.xml
	rm -rf logs
