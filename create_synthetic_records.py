# %%
def create_synthetic_records(n_records,n_duplicates0 = 0,n_duplicates1 = 0,n_duplicates2 = 0,n_duplicates3 = 0 ):

    from faker import Faker
    import pandas as pd
    import numpy as np
    import random

    # Instantiate Faker
    fake = Faker() 

    # Create a new DataFrame for storing the original records and their duplicates
    df_new = pd.DataFrame({
        'client_id': range(1, n_records+1),
        'client_name': [fake.name() for _ in range(n_records)],
        'address': [fake.address().replace('\n', ', ') for _ in range(n_records)],
        'email': [fake.email() for _ in range(n_records)],
        'phone_number': [fake.phone_number() for _ in range(n_records)],
        'duplicate_id': range(1, n_records+1),  # Add a new column for the duplicate ID
        'num_modifications': [0] * n_records  # Add a new column for the number of modifications
    })

    # Define modification functions
    def modify_name(name):
        mods = ['add_middle_initial'
                ,'add_prefix'
                ,'typo'
                ,'switch_name_order'
                ,'change_case']
        mod = random.choice(mods)
        print(f"Chosen modification: {mod}")
        
        if mod == 'add_middle_initial':
            if len(name.split()) > 1:
                first, last = name.split(' ')
                middle_initial = fake.random_uppercase_letter()
                modified_name = f"{first} {middle_initial}. {last}"
            else:
                modified_name = f"{name} {fake.random_uppercase_letter}."
                
        elif mod == 'add_prefix':
            prefix = fake.prefix()
            modified_name = f"{prefix} {name}"
            
        elif mod == 'typo':
            index = random.randint(0, len(name)-1)
            modified_name = name[:index] + fake.random_lowercase_letter() + name[index+1:]
            
        elif mod == 'switch_name_order':
            name_parts = name.split()
            modified_name = ', '.join(name_parts[::-1])
            
        elif mod == 'change_case':
            modified_name = name.upper() if random.choice([True, False]) else name.lower()
        
        print(f"Modified name: {modified_name}")
        return modified_name


    def modify_email(email):
        user, domain = email.split('@')
        new_domain = fake.free_email_domain()
        return f"{user}@{new_domain}"

    def modify_address(address):
        new_address = fake.street_address()
        city_state_zip = address.split(', ')[1:]
        return f"{new_address}, {', '.join(city_state_zip)}"

    def modify_phone_number(phone_number): 
        digits = [char for char in phone_number if char.isdigit()]
        digit_to_replace = random.choice(digits)
        return phone_number.replace(digit_to_replace, str(fake.random_digit()), 1)

    def modification(field, value):
        if field == 'client_name':
            return modify_name(value)
        elif field == 'address':
            return modify_address(value)
        elif field == 'email':
            return modify_email(value)
        elif field == 'phone_number':
            return modify_phone_number(value)

    # Make slight modifications to create duplicates
    for idx in range(n_duplicates0):

        #print("duplicate #: ", idx)
        # Create the first duplicate
        duplicate = df_new.loc[idx].copy()
        #print(duplicate)
        new_client_id = df_new['client_id'].max() + 1  # Assign a new client_id
        duplicate['client_id'] = new_client_id
        duplicate['duplicate_id'] = int(idx + 1)  # Assign the duplicate ID to be the original client_id

        # Randomly select zero or more fields to modify (excluding client_id)
        num_modifications = np.random.choice([0, 0, 1, 1, 2])  # Make 2 modifications rarer
        duplicate['num_modifications'] = num_modifications
        if num_modifications > 0:
            fields_to_modify = np.random.choice(['client_name', 'address', 'email', 'phone_number'], size=num_modifications, replace=False)

            for field in fields_to_modify:
                # Apply the modification
                value = duplicate[field]
                duplicate[field] = modification(field,value)

        # Append the duplicate to the DataFrame
        df_new = df_new.append(duplicate, ignore_index=True)
        #print(idx,duplicate)

        # Create additional duplicates for a subset of the original records
        if idx < n_duplicates1:
            # Create one more duplicate for the first 50 original records
            duplicate = df_new.loc[idx].copy()
            new_client_id = df_new['client_id'].max() + 1  # Assign a new client_id
            duplicate['client_id'] = new_client_id
            duplicate['duplicate_id'] = int(idx + 1)  # Assign the duplicate ID to be the original client_id

            num_modifications = np.random.choice([0, 0, 1, 1, 2])  # Make 2 modifications rarer
            duplicate['num_modifications'] = num_modifications
            if num_modifications > 0:
                fields_to_modify = np.random.choice(['client_name', 'address', 'email', 'phone_number'], size=num_modifications, replace=False)

                for field in fields_to_modify:
                    # Apply the modification
                    value = duplicate[field]
                    duplicate[field] = modification(field,value)

            # Append the duplicate to the DataFrame
            df_new = df_new.append(duplicate, ignore_index=True)
            #print(idx,duplicate)

        if idx < n_duplicates2:
            # Create more duplicates
            duplicate = df_new.loc[idx].copy()
            new_client_id = df_new['client_id'].max() + 1  # Assign a new client_id
            duplicate['client_id'] = new_client_id
            duplicate['duplicate_id'] = int(idx + 1)  # Assign the duplicate ID to be the original client_id

            num_modifications = np.random.choice([0, 0, 1, 1, 2])  # Make 2 modifications rarer
            duplicate['num_modifications'] = num_modifications
            if num_modifications > 0:
                fields_to_modify = np.random.choice(['client_name', 'address', 'email', 'phone_number'], size=num_modifications, replace=False)

                for field in fields_to_modify:
                    # Apply the modification
                    value = duplicate[field]
                    duplicate[field] = modification(field,value)

            # Append the duplicate to the DataFrame
            df_new = df_new.append(duplicate, ignore_index=True)
            #print(idx,duplicate)

        if idx == 0:
            # Create more duplicates for the first original record
            for _ in range(n_duplicates3):
                duplicate = df_new.loc[idx].copy()
                new_client_id = df_new['client_id'].max() + 1  # Assign a new client_id
                duplicate['client_id'] = new_client_id
                duplicate['duplicate_id'] = int(idx + 1)  # Assign the duplicate ID to be the original client_id

                num_modifications = np.random.choice([0, 0, 1, 1, 2])  # Make 2 modifications rarer
                duplicate['num_modifications'] = num_modifications
                if num_modifications > 0:
                    fields_to_modify = np.random.choice(['client_name', 'address', 'email', 'phone_number'], size=num_modifications, replace=False)

                    for field in fields_to_modify:
                        # Apply the modification
                        value = duplicate[field]
                        duplicate[field] = modification(field,value)

                # Append the duplicate to the DataFrame
                df_new = df_new.append(duplicate, ignore_index=True)
                #print(idx,duplicate)
        
        #print(len(df_new))

    # Shuffle the DataFrame
    df_new = df_new.sample(frac=1).reset_index(drop=True)
    return df_new

# %%



