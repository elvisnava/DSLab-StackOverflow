# How to import a StackExchange dump into a Postgres DB

1. Install `postgresql` and relevant libraries with  
  `sudo apt install postgresql postgresql-contrib libpq-dev `

2. Create a role for the current user and create the db:  
  Log into the *postgres* user  
  `sudo -i -u postgres`  
  Create a role for your user account  
  `createuser --interactive`    
  At the prompt, write your personal user name, same as the one on your machine  
  `Enter name of role to add: YOUR_USERNAME`  
  `Shall the new role be a superuser? (y/n) y`  
  Create the *crossvalidated* db  
  `createdb crossvalidated`  
  Press CTRL+D to exit the *postgres* user  
  To enter the db from now on you can use the command  
  `psql -d crossvalidated`

3. Install `psycopg2` to query the db from python, and other prerequisites  
  `pip3 install lxml psycopg2 libarchive-c`

4. Navigate outside the DSLab repo and clone the `stackexchange-dump-to-postgres` repo  
  `git clone https://github.com/Networks-Learning/stackexchange-dump-to-postgres.git`

5. Enter the repo directory and copy the files `Badges.xml`, `Votes.xml`, `Posts.xml`, `Users.xml`, `Tags.xml` into it.

6. Run the commands  
  `python3 load_into_pg.py -t Posts -f Posts.xml -d crossvalidated --with-post-body`  
  `python3 load_into_pg.py -t Tags -f Tags.xml -d crossvalidated`  
  `python3 load_into_pg.py -t Users -f Users.xml -d crossvalidated`  
  `python3 load_into_pg.py -t Votes -f Votes.xml -d crossvalidated`  
  `python3 load_into_pg.py -t Badges -f Badges.xml -d crossvalidated`  
  `psql -d crossvalidated < ./sql/final_post.sql`

# Make the Postgres db accessible from localhost without a password

1. Find where the `pg_hba.conf` file is by running  
  `psql -d postgres -c 'SHOW hba_file;'`

2. Edit the `pg_hba.conf` file by writing `trust` instead of `md5` in lines  
  `# IPv4 local connections:`  
  `host    all             all             127.0.0.1/32            md5`  
  `# IPv6 local connections:`  
  `host    all             all             ::1/128                 md5`

3. Restart the postgres service  
  `sudo service postgresql restart`

4. Use the address `'postgresql://localhost/crossvalidated'` with sqlalchemy
