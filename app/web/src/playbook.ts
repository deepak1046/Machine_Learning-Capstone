/** General first-response guidance per attack vector (what an operator would check; nothing here is applied automatically). */
export const PLAYBOOK: Record<string, string> = {
  DNS: "DNS reflection/amplification. Drop or rate-limit unsolicited UDP/53 responses at the edge and ask the upstream provider to filter spoofed sources.",
  LDAP: "CLDAP reflection. UDP/389 should not be reachable from the internet; block it at the perimeter.",
  MSSQL: "SQL Server Resolution Service reflection. Block UDP/1434 from untrusted networks.",
  NTP: "NTP amplification. Block or rate-limit UDP/123 replies; reflectors should disable the monlist command.",
  NetBIOS: "NetBIOS name-service reflection. Block UDP/137 at the perimeter.",
  Portmap: "Portmapper (RPC) reflection. Block UDP/111 at the perimeter.",
  SNMP: "SNMP reflection. Keep UDP/161 off the internet and limit it to management networks.",
  SSDP: "SSDP/UPnP reflection. Block UDP/1900 at the perimeter.",
  Syn: "SYN flood. Enable SYN cookies, cap half-open connections per source, and bring in upstream scrubbing if the link saturates.",
  TFTP: "TFTP reflection. Block UDP/69 from the internet.",
  UDP: "Generic UDP flood. Rate-limit UDP per source and towards unused ports; engage upstream scrubbing if volume keeps growing.",
  UDPLag: "UDP-lag flood, aimed at degrading real-time sessions. Rate-limit UDP per source and ask the upstream provider to filter.",
  "Reflection-Amp":
    "Reflection/amplification (DNS, LDAP or SNMP). Block or rate-limit UDP/53, 389 and 161 replies at the edge.",
  "NetBIOS-Portmap": "NetBIOS or Portmap reflection. Block UDP/137 and UDP/111 at the perimeter.",
  "UDP-Family":
    "UDP flood or SSDP reflection. Rate-limit UDP per source and block UDP/1900 at the perimeter.",
};

export const FALLBACK_ADVICE = "Rate-limit the offending traffic at the edge and contact the upstream provider.";
