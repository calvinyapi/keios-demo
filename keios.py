import mesop as me

def load(e: me.LoadEvent):
  me.set_theme_mode("system")



@me.page(
  on_load=load,
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://mesop-dev.github.io"]
  ),
  path="/keios_starter_kit",
)
def starter_kit():
  me.image(
    src="https://i.postimg.cc/XJr9Bd0p/explo.png",
    style=me.Style(width="20%"),
  )
  me.text("Keios Starter Kit")
  me.markdown("Bienvenue dans le kit de démarrage Keios !")

  me.markdown("### Étape 1 : Configuration du flux vidéo")
  me.markdown("Assurez-vous que votre caméra Tasmota est configurée pour diffuser un flux MJPEG.")
  me.markdown("Utilisez l'URL suivante pour le flux : `http://<adresse_ip_de_votre_camera>:81/stream`")
  with me.box():
        me.html('<img src="http://192.168.1.50:81/stream" style="max-width:100%;" />')
