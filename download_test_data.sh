#!/bin/bash

set -eo pipefail

if [ -z "$DATA_PASSWORD" ]; then
	echo -n "Password: "
	read -s DATA_PASSWORD
	echo
fi

URL=https://nextcloud.e5.physik.tu-dortmund.de/public.php/webdav/

curl -sSfL -o data.zip -u "pffcqqxsLbZPCc7:$DATA_PASSWORD" "$URL"
unzip data.zip
rm data.zip
