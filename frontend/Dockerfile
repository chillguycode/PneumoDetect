FROM nginx:stable-alpine


RUN rm /etc/nginx/conf.d/default.conf

#Copy our custom Nginx Config
COPY frontend/nginx.conf  /etc/nginx/conf.d/default.conf

COPY frontend/.  /usr/share/nginx/html
