import requests
import json

def get_customer_history_context(customer_id: str) -> str:
    """Get customer history context from the API"""
    url = 'https://agentapi.ifbanalytics.com/CustomerDetailsInnerCols_SB'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "customer": customer_id,
        "request_columns": {
            "crm_init":["zzpurchase_date","zzinstall_date","zzpost_code1","city1","zz0010","zz0012","zzsubcat","ZZPROD_DESC","zzr3ser_no","warrantydesc","warranty_sdate","warranty_edate"],
            "crm_allcall": ["Ticket","CallType","Status","Product","PostingDate","ClosedTime","ServiceType","MachineStatus","Medium","Origin"],
            "cust_likes": ["textbox","timestamp"],
            "sap_spu": ["MATDES","CRMTICKET","MACHSTAT","SPARE","QUANTITY"],
            "crm_amccontracts": ["Srno","Amctype","Cont_strt_dat","Cont_end_dat","zzmat_grp","warconv","price"]
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        
        # Handle 404 and other error status codes
        if response.status_code != 200:
            return "No customer history found."
            
        data = response.json()
        
        history_parts = []

        # CRM INIT
        for item in data.get('crm_init', []):
            history_parts.append(
                f"Product: {item['ZZPROD_DESC']}, Installed on: {item['zzinstall_date']}, "
                f"City: {item['city1']}, Serial: {item['zzr3ser_no']}, "
                f"Warranty: {item['warrantydesc']} ({item['warranty_sdate']} to {item['warranty_edate']})"
            )

        # CRM ALL CALL
        if data.get('crm_allcall'):
            for call in data['crm_allcall']:
                history_parts.append(
                    f"Service Ticket {call['Ticket']} for {call['Product']} was {call['Status']} "
                    f"on {call['PostingDate']}. Machine Status: {call['MachineStatus']}"
                )

        # Likes
        if data.get('cust_likes'):
            for like in data['cust_likes']:
                history_parts.append(f"Customer mentioned: '{like['textbox']}' on {like['timestamp']}")

        # AMC Contracts
        for amc in data.get('crm_amccontracts', []):
            history_parts.append(
                f"AMC Type: {amc['Amctype']}, Valid from {amc['Cont_strt_dat']} to {amc['Cont_end_dat']}, "
                f"Group: {amc['zzmat_grp']}, Price: {amc['price']}"
            )

        # SPU History
        for spu in data.get('sap_spu', []):
            history_parts.append(
                f"Spare Used: {spu['SPARE']} (Qty: {spu['QUANTITY']}) in ticket {spu['CRMTICKET']} "
                f"for {spu['MATDES']}. Machine Status: {spu['MACHSTAT']}"
            )

        return "\n".join(history_parts) if history_parts else "No significant customer history found."
        
    except requests.exceptions.RequestException as e:
        # Handle any request-related errors gracefully
        return "No customer history found."

if __name__ == "__main__":
    print(get_customer_history_context("23784201"))