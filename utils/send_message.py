

from email.mime.text import MIMEText
import smtplib
import requests


# 邮箱信息
EMAIL_MSG_IDENTIFY = 'abc'
EMAIL_SEND_ADDR = 'abc@qq.com'
EMAIL_TO = 'abc@outlook.com'

# 微信信息
SECRET = '123'
CORP_ID = '123'
AGENT_ID = 123

def send_email(content):
    msg_to = EMAIL_TO

    subject = '量化日志邮件'

    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SEND_ADDR
    msg['To'] = msg_to

    try:
        s = smtplib.SMTP_SSL('smtp.qq.com', 465)
        s.login(EMAIL_SEND_ADDR, EMAIL_MSG_IDENTIFY)
        s.sendmail(EMAIL_SEND_ADDR, msg_to, msg.as_string())
    except BaseException as e:
        print('邮件发送失败：'+str(e))
    finally:
        s.quit()


def get_token():
    url = 'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={}&corpsecret={}'.format(CORP_ID, SECRET)
    resp = requests.get(url)
    token = resp.json().get('access_token')
    return token


def get_user_id_but_phone(phone_no):
    token = get_token()
    url = 'https://qyapi.weixin.qq.com/cgi-bin/user/getuserid?access_token={}'.format(token)
    data = {"mobile": phone_no}
    resp = requests.post(url=url, json=data)
    print(resp.text)

def send_wechat_msg(content, user_id='huangzheng'):
    token = get_token()
    url = 'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={}'.format(token)
    data = {
            "touser" : user_id,
            "msgtype" : "text",
            "agentid" : AGENT_ID,
            "text" : {
                "content" : content
            },
            "safe":0,
            "enable_id_trans": 0,
            "enable_duplicate_check": 0,
            "duplicate_check_interval": 1800
            }
    resp = requests.post(url=url, json=data)
    if resp.json().get('errcode') == 0:
        print('微信消息发送成功')
        return True
    else:
        print('！！！微信发送失败')
        return False


if __name__ == '__main__':
    send_email('这个是日志邮件')