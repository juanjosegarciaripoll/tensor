LOGS_DIR = $(HOME)/src/tensor_logs

www: test.html logs

test.html: build_test_page joint_report.xml
	xsltproc -x $(top_srcdir)/joint_report.xsl joint_report.xml > $@

joint_report.xml:
	echo '<?xml version="1.0" encoding="UTF-8"?>' > $@
	for i in $(LOGS_DIR)/*/report.xml; do \
	  dir=`dirname $$i`; \
	  name=`basename $$dir`; \
	  nfailures=`cat $${dir}/nfailures`; \
	  if [ "x$${nfailures}" = "x" ]; then nfailures=1; fi; \
	  sed  -e "/?xml/d;s/testframe/testframe name='$${name}' nfailures='$$nfailures'" $i >> $@ ; \
	done

logs: $(LOGS_DIR)
	if [ -d logs ]; rm -f logs/*.html ; else mkdir logs; fi
	for i in $(LOGS_DIR)/*/report.xml; do \
	  dir=`dirname $$i`; \
	  name=`basename $${dir}`; \
	  xsltproc -x $(top_srcdir)/report.xsl $i > logs/$name; \
	done
