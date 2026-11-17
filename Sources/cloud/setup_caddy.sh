#!/usr/bin/env bash
# Run ONCE on the Google Cloud machine (Ubuntu). Installs Caddy, which gets a free HTTPS certificate
# and passes requests on to the federated-learning server on port 8000 of the same machine.
#
#   bash setup_caddy.sh             finds the machine's external IP by itself
#   bash setup_caddy.sh 34.1.2.3    or give it the external IP
#
# The address becomes https://34-1-2-3.sslip.io (sslip.io turns the name back into the IP for free).
set -euo pipefail

IP="${1:-}"
if [ -z "$IP" ]; then
  IP=$(curl -s -m 5 -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip" || true)
fi
if ! [[ "$IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Could not find the external IP. Copy it from the VM list and run:  bash setup_caddy.sh <EXTERNAL_IP>"
  exit 1
fi
NAME="${IP//./-}.sslip.io"
URL="https://$NAME"

if ! command -v caddy >/dev/null 2>&1; then
  echo "Installing Caddy..."
  sudo apt-get update -q
  sudo apt-get install -y -q caddy
fi

sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
# HTTPS address of the federated-learning server. Caddy gets and renews the certificate itself.
$NAME {
    reverse_proxy 127.0.0.1:8000
}
CADDY
sudo systemctl enable --now caddy >/dev/null 2>&1
sudo systemctl reload caddy
echo "$URL" > "$HOME/.posefl_url"

echo "Waiting for the HTTPS certificate (up to a minute)..."
for _ in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$URL/health" || true)
  # 401/502/200 all mean HTTPS works (502 = the server is not started yet)
  if [ "$code" = "200" ] || [ "$code" = "401" ] || [ "$code" = "502" ]; then
    echo
    echo "HTTPS is working:  $URL"
    echo "Next:  bash start_server.sh YOUR_PASSWORD"
    exit 0
  fi
  sleep 2
done
echo
echo "Caddy is set up for $URL but the certificate is not ready yet."
echo "Check that 'Allow HTTP traffic' and 'Allow HTTPS traffic' are ticked for this VM,"
echo "then look at the log with:  sudo journalctl -u caddy --no-pager | tail -30"
exit 1
