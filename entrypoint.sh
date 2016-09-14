#!/bin/bash
function generate_passwd_file() {
  export USER_ID=$(id -u)
  export GROUP_ID=$(id -u)
  echo "openface:x:${USER_ID}:${GROUP_ID}:openface:${HOME}:/bin/bash" >> /var/lib/extrausers/passwd
}

generate_passwd_file
id
cat /var/lib/extrausers/passwd
ls -l /root/openface/demos/web/start-servers.sh

bash -x /root/openface/demos/web/start-servers.sh
