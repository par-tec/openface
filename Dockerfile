FROM bamos/openface
MAINTAINER Roberto Polli <roberto.polli@par-tec.it>

# Use libnss-wrapper to allow openshift 3.x user scrambling. 
RUN apt-get  update && apt-get -y install libnss-extrausers && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN sed -i 's/passwd:         compat/passwd:         compat extrausers/' /etc/nsswitch.conf
RUN cat /etc/passwd  > /var/lib/extrausers/passwd
RUN touch  /var/lib/extrausers/passwd && chmod a+w /var/lib/extrausers/passwd

# Install flask for further web stuff
RUN pip install Flask-Uploads

# Update the webapp code with the checked out code
ADD ./demos/web /root/openface/demos/web

# To run as a generic user, we need to chmod 777
RUN chmod -R 777 /root
#RUN chmod +x /root/openface/demos/web/*.py
#RUN chmod +x /root/openface/demos/web/*.sh

# this entrypoint.sh adds the generic `openface` user in /var/lib/extrausers/passwd 
ADD ./entrypoint.sh /entrypoint.sh
RUN chmod +x  /entrypoint.sh
USER 1001

EXPOSE 8000 9000 5000
CMD /bin/bash -l -c '/entrypoint.sh'
