from __future__ import annotations

TAXONOMY = {
    "Functional Correctness": [
        "Safety", "Liveness", "Reachability", "Timing Constraints", "Hybrid Dynamics"
    ],
    "Information Protection": [
        "Confidentiality", "Integrity", "Availability", "Authenticity", "Authorization",
        "Accountability", "Non-repudiation"
    ],
    "Operational Assurance": [
        "Privacy", "Reliability", "Resilience", "Recoverability", "Compliance", "Explainability"
    ],
}

PARENT = {p: parent for parent, props in TAXONOMY.items() for p in props}
PROPERTIES = [p for props in TAXONOMY.values() for p in props]

# These statements are intentionally consequence-based. The model must test whether
# the incident text entails the actual property violation; attack type alone is insufficient.
PROPERTY_SPECS = {
    "Safety": {
        "retrieval": "hazardous unsafe physical state, unsafe control decision, dangerous physical consequence",
        "confirmed": "The incident actually caused the cyber-physical system to enter a hazardous or unsafe physical state.",
        "potential": "The incident could potentially create a hazardous or unsafe physical state, but no actual unsafe state was confirmed.",
        "unaffected": "The source explicitly states that no hazardous or unsafe physical consequence occurred.",
        "claimed": "A source claimed that an unsafe physical consequence occurred, but that claim was not independently confirmed.",
    },
    "Liveness": {
        "retrieval": "required progress never completes, command or task indefinitely blocked, normal progression prevented",
        "confirmed": "The incident actually prevented a required system task, command, process, or recovery action from eventually completing.",
        "potential": "The incident could prevent required system progress or completion, but no actual liveness failure was confirmed.",
        "unaffected": "The source explicitly states that required system progress and completion were not prevented.",
        "claimed": "A source claimed that required system progress was prevented, but the claim was not independently confirmed.",
    },
    "Reachability": {
        "retrieval": "unauthorized state reached, attacker gained privileged access, commands reached controller, malware crossed segmentation",
        "confirmed": "The incident actually reached an unauthorized system state, privilege, controller, network zone, or protected component.",
        "potential": "The incident could make an unauthorized state or protected component reachable, but actual reachability was not confirmed.",
        "unaffected": "The source explicitly states that protected states, controllers, privileges, or segmented zones were not reached.",
        "claimed": "A source claimed unauthorized reachability or access occurred, but the claim was not independently confirmed.",
    },
    "Timing Constraints": {
        "retrieval": "missed deadline, excessive latency, delayed emergency response, stale timing, synchronization failure",
        "confirmed": "The incident actually violated a required timing bound, deadline, synchronization requirement, or time-critical response.",
        "potential": "The incident could violate a timing bound or deadline, but no actual timing violation was confirmed.",
        "unaffected": "The source explicitly states that timing bounds, deadlines, or synchronization were not affected.",
        "claimed": "A source claimed a timing violation occurred, but the claim was not independently confirmed.",
    },
    "Hybrid Dynamics": {
        "retrieval": "cyber manipulation changed continuous physical dynamics, control loop, trajectory, pressure, flow, frequency, physical evolution",
        "confirmed": "The incident actually manipulated or destabilized the continuous physical dynamics of the cyber-physical process.",
        "potential": "The incident could affect continuous physical dynamics, but no actual change in physical evolution was confirmed.",
        "unaffected": "The source explicitly states that continuous physical dynamics or the physical process were not affected.",
        "claimed": "A source claimed continuous physical dynamics were manipulated, but the claim was not independently confirmed.",
    },
    "Confidentiality": {
        "retrieval": "sensitive information disclosed, exposed, stolen, exfiltrated, leaked to unauthorized party",
        "confirmed": "The incident actually disclosed, exposed, leaked, or exfiltrated sensitive information to an unauthorized party.",
        "potential": "Sensitive information could have been exposed, but no actual unauthorized disclosure was confirmed.",
        "unaffected": "The source explicitly states that sensitive information was not accessed, disclosed, exposed, or stolen.",
        "claimed": "A party claimed sensitive information was stolen or exposed, but the claim was not independently confirmed.",
    },
    "Integrity": {
        "retrieval": "unauthorized modification of data, sensor values, commands, PLC logic, firmware, software, parameters",
        "confirmed": "The incident actually caused unauthorized modification, falsification, or corruption of data, commands, software, firmware, logic, or configuration.",
        "potential": "The incident could modify or falsify data, commands, software, firmware, logic, or configuration, but no actual modification was confirmed.",
        "unaffected": "The source explicitly states that data, commands, software, firmware, logic, or configuration integrity was not affected.",
        "claimed": "A party claimed unauthorized modification or corruption occurred, but the claim was not independently confirmed.",
    },
    "Availability": {
        "retrieval": "authorized access unavailable, service outage, system inaccessible, denial of service, lockout, operational system unavailable",
        "confirmed": "The incident actually made a required system, service, controller, application, or communication resource unavailable to authorized users or processes.",
        "potential": "The incident could make a required resource unavailable, but no actual availability loss was confirmed.",
        "unaffected": "The source explicitly states that required systems, services, or operations remained available and were not disrupted.",
        "claimed": "A party claimed a required system or service became unavailable, but the claim was not independently confirmed.",
    },
    "Authenticity": {
        "retrieval": "impersonation, spoofed identity, forged credentials, fake sensor, malicious device masquerading as trusted",
        "confirmed": "The incident actually involved successful impersonation, spoofing, forged identity, or a malicious entity being accepted as authentic.",
        "potential": "The incident could enable impersonation or spoofing, but no successful authenticity violation was confirmed.",
        "unaffected": "The source explicitly states that identities or communicating entities were not spoofed or impersonated.",
        "claimed": "A party claimed impersonation or spoofing occurred, but the claim was not independently confirmed.",
    },
    "Authorization": {
        "retrieval": "unauthorized action, privilege escalation, excessive permission, unauthorized command, unauthorized configuration change",
        "confirmed": "The incident actually allowed an entity to perform an action or operation that it was not authorized to perform.",
        "potential": "The incident could permit unauthorized actions or excessive privileges, but no actual authorization violation was confirmed.",
        "unaffected": "The source explicitly states that unauthorized actions or privilege misuse did not occur.",
        "claimed": "A party claimed unauthorized actions or privilege misuse occurred, but the claim was not independently confirmed.",
    },
    "Accountability": {
        "retrieval": "audit trail missing, logs deleted, actions cannot be attributed, forensic attribution impossible, logging failure",
        "confirmed": "The incident actually prevented important actions from being attributable because logging, audit trails, or event records were missing, destroyed, or untrustworthy.",
        "potential": "The incident could impair attribution or auditability, but no actual accountability failure was confirmed.",
        "unaffected": "The source explicitly states that trustworthy logs, audit trails, and attribution remained available.",
        "claimed": "A party claimed logs or attribution were compromised, but the claim was not independently confirmed.",
    },
    "Non-repudiation": {
        "retrieval": "denial of performed action, missing cryptographic proof, signatures invalid, command history cannot prove who acted",
        "confirmed": "The incident actually undermined cryptographic or audit evidence needed to prove that a participant performed a specific action.",
        "potential": "The incident could undermine proof of performed actions, but no actual non-repudiation failure was confirmed.",
        "unaffected": "The source explicitly states that signatures, authenticated records, or proof of actions remained valid.",
        "claimed": "A party claimed proof of performed actions was undermined, but the claim was not independently confirmed.",
    },
    "Privacy": {
        "retrieval": "personal data, medical records, location, behavioral data, PII, improper use of personal information",
        "confirmed": "The incident actually violated the privacy of individuals through unauthorized exposure, use, processing, or dissemination of personal or sensitive personal information.",
        "potential": "Personal information could have been exposed or misused, but no actual privacy violation was confirmed.",
        "unaffected": "The source explicitly states that personal or sensitive personal information was not exposed, accessed, or misused.",
        "claimed": "A party claimed personal information was exposed or misused, but the claim was not independently confirmed.",
    },
    "Reliability": {
        "retrieval": "system failed to perform intended function consistently, unstable operation, repeated malfunction, dependable operation degraded",
        "confirmed": "The incident actually caused the system to fail to perform its intended function consistently and dependably under expected operation.",
        "potential": "The incident could reduce dependable operation, but no actual reliability degradation was confirmed.",
        "unaffected": "The source explicitly states that the system continued to perform its intended functions reliably.",
        "claimed": "A party claimed reliability was degraded, but the claim was not independently confirmed.",
    },
    "Resilience": {
        "retrieval": "essential function could not be maintained during attack, graceful degradation failed, continuity under disruption failed",
        "confirmed": "The incident actually demonstrated failure to withstand the disruption while maintaining essential functionality.",
        "potential": "The incident could challenge resilience, but no actual failure to maintain essential functionality was confirmed.",
        "unaffected": "The source explicitly states that essential functionality was maintained despite the disruption.",
        "claimed": "A party claimed essential functionality could not be maintained, but the claim was not independently confirmed.",
    },
    "Recoverability": {
        "retrieval": "restoration delayed, recovery failed, prolonged restoration, rebuild, restore from backup, services not restored",
        "confirmed": "The incident actually impaired or substantially delayed restoration of normal trusted operation after the disruption.",
        "potential": "The incident could complicate restoration, but no actual recoverability problem was confirmed.",
        "unaffected": "The source explicitly states that normal trusted operation was restored without a material recovery problem.",
        "claimed": "A party claimed recovery was impaired or delayed, but the claim was not independently confirmed.",
    },
    "Compliance": {
        "retrieval": "regulatory requirement violated, mandatory control failure, legal compliance breach, required standard not met",
        "confirmed": "The incident actually demonstrated violation of an applicable regulatory, legal, or mandatory security or operational requirement.",
        "potential": "The incident could create a compliance issue, but no actual regulatory or mandatory requirement violation was confirmed.",
        "unaffected": "The source explicitly states that applicable regulatory or mandatory requirements remained satisfied.",
        "claimed": "A party claimed a compliance violation occurred, but the claim was not independently confirmed.",
    },
    "Explainability": {
        "retrieval": "system decision could not be explained, opaque automated decision, lack of interpretable rationale, operator cannot understand decision",
        "confirmed": "The incident actually revealed that a security-relevant or operational system decision could not be meaningfully explained or interpreted by responsible humans.",
        "potential": "The incident could create an explainability problem, but no actual inability to explain a relevant decision was confirmed.",
        "unaffected": "The source explicitly states that relevant system decisions remained interpretable and explainable.",
        "claimed": "A party claimed relevant decisions were not explainable, but the claim was not independently confirmed.",
    },
}
